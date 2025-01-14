from typing import Tuple
import pandas as pd
import numpy as np
import re
import os
from math import floor, ceil

import dp_policy.config as config
from dp_policy.titlei.mechanisms import Sampled, Mechanism

from typing import Tuple


def get_official_combined(path: str) -> pd.DataFrame:
    """Load official Dept Ed data.

    Args:
        path (str): Path of file.

    Returns:
        pd.DataFrame: Return cleaned dataframe.
    """
    allocs = get_official(
        path, "Allocations", 10, [
            "Sort C",
            "State FIPS Code",
            "State",
            "LEAID",
            "Name",
            "Basic Hold Harmless",
            "Concentration Hold Harmless",
            "Targeted Hold Harmless",
            "EFIG Hold Harmless",
            "Total Hold Harmless",
            "Basic Alloc",
            "Concentration Alloc",
            "Targeted Alloc",
            "EFIG Alloc",
            "Total Alloc",
            "Hold Harmless Percentage",
            "Resident Pop."
        ]
    )
    counts = get_official(
        path, "Populations", 7, [
            "Sort C",
            "State FIPS Code",
            "State",
            "LEAID",
            "Name",
            "Total Formula Count",
            "5-17 Pop.",
            "Percent Formula",
            "Basic Eligibles",
            "Concentration Eligibles",
            "Targeted Eligibles",
            "Weighted Counts Targeted",
            "Weighted Counts EFIG"
        ]
    )
    combined = allocs.set_index("LEAID").join(
        counts.drop(columns=[
            "Sort C",
            "State FIPS Code",
            "State",
            "Name"
        ]).set_index("LEAID"),
        how="inner"
    ).reset_index()
    combined.loc[:, "State FIPS Code"] = \
        combined["State FIPS Code"].astype(int)
    combined["District ID"], _ = split_leaids(combined["LEAID"].astype(int))
    return combined.set_index(["State FIPS Code", "District ID"])


def get_official(path, sheet, header, columns):
    allocs = pd.read_excel(path, sheet_name=sheet, header=header)
    allocs = allocs.iloc[1:, :len(columns)]
    allocs.columns = columns
    return allocs


def get_saipe(path: str) -> pd.DataFrame:
    """Get district-level SAIPE data.

    Args:
        path (str): Path to file.

    Returns:
        pd.DataFrame: Cleaned district-level data.
    """
    saipe = pd.read_excel(path, header=2)\
        .set_index(["State FIPS Code", "District ID"])
    saipe["cv"] = saipe.apply(
        lambda x: median_cv(x["Estimated Total Population"]),
        axis=1
    )
    return saipe


def get_county_saipe(path: str):
    """Get county-level SAIPE data.

    Args:
        path (str): Path to file.

    Returns:
        pd.DataFrame: Cleaned county-level data.
    """
    saipe = pd.read_excel(path, header=3, usecols=[
        "State FIPS Code",
        "County FIPS Code",
        "Name",
        "Poverty Estimate, All Ages",
        "Poverty Percent, All Ages",
        "Poverty Estimate, Age 5-17 in Families",
        "Poverty Percent, Age 5-17 in Families"
    ]).replace('.', np.NaN).fillna(0.0, )

    # convert county FIPS codes to district ids
    saipe["District ID"] = saipe["County FIPS Code"]
    # convert to saipe district column names
    saipe["Estimated Total Population"] = \
        saipe["Poverty Estimate, All Ages"].astype(float) \
        / (saipe["Poverty Percent, All Ages"].astype(float) / 100)
    saipe["Estimated Population 5-17"] = \
        saipe["Poverty Estimate, Age 5-17 in Families"].astype(float) \
        / (saipe["Poverty Percent, Age 5-17 in Families"].astype(float) / 100)
    saipe[
        'Estimated number of relevant children 5 to 17 years old '
        'in poverty who are related to the householder'
    ] = saipe["Poverty Estimate, Age 5-17 in Families"]

    saipe["cv"] = saipe.apply(
        lambda x: median_cv(x["Estimated Total Population"]),
        axis=1
    )

    return saipe.set_index(["State FIPS Code", "District ID"]).drop(columns=[
        "Poverty Estimate, All Ages",
        "Poverty Percent, All Ages",
        "Poverty Estimate, Age 5-17 in Families",
        "Poverty Percent, Age 5-17 in Families"
    ])


def district_id_from_name(df, name, state=None):
    if state:
        df = df.loc[state, :]
    ind = df[df["Name"] == name].index.get_level_values("District ID")
    if len(ind) == 0:
        raise Exception("No districts with the name", name)
    if len(ind) > 1:
        raise Exception("Multiple district IDs with the name", name)
    return ind[0]


def get_inputs(
    year,
    baseline="prelim",
    avg_lag=0,
    verbose=True,
    use_official_children=False
):
    """Load the inputs for calculating title i allocations

    Args:
        year (int): _description_
        baseline (str, optional): what official dep ed file to use as a
            baseline. Options are "prelim", "final", and "revfinal." Defaults
            to "prelim".
        avg_lag (int, optional): Whether to average, and by how many years.
            Defaults to 0.
        verbose (bool, optional): Defaults to True.
        use_official_children (bool, optional): Whether to use the official
            # total children from Dep Ed, instead of the SAIPE # of children.
            # of children in poverty will always be from SAIPE. Designed for
            validation purposes. Defaults to False.

    Returns:
        pd.DataFrame: combined dataframe of inputs for use in an allocator.
    """
    # official ESEA data
    if year < 2020:
        print("[WARN] Using official data for 2020 instead.")
        official_year = 2020
    else:
        official_year = year

    print(config.root)
    official = get_official_combined(os.path.join(
        config.root,
        f"data/titlei-allocations/{baseline}_{str(official_year)[2:]}.xls"
    )).drop(columns=[
        'LEAID',
        'Sort C',
        'State',
        'Resident Pop.',
        # 'Name'
    ])
    official.columns = [
        f"official_{c.lower().replace(' ', '_')}" if c != 'Name' else c
        for c in official.columns
    ]

    # join with Census SAIPE
    if avg_lag > 0:
        saipe, county_saipe = average_saipe(year-2, avg_lag, verbose=verbose)
    else:
        saipe = get_saipe(f"{config.root}/data/saipe{str(year-2)[2:]}.xls")
        county_saipe = get_county_saipe(
            f"{config.root}/data/county_saipe{str(year-2)[2:]}.xls"
        )
    # for some reason, NY naming convention different...
    # fixing that here
    county_saipe.rename(index={
        district_id_from_name(county_saipe, c, 36):
            district_id_from_name(official, c, 36)
        for c in [
            "Bronx County",
            "Kings County",
            "New York County",
            "Queens County",
            "Richmond County"
        ]
    }, level='District ID', inplace=True)
    saipe_stacked = impute_missing(saipe, county_saipe, verbose=verbose)

    if verbose:
        print("-- WARNING: dropping some balances from total budget --")
        for name, filter in [
            ("Puerto Rico", official.Name == "Puerto Rico"),
            ("County balances", official.Name.str.contains("BALANCE OF")),
            ("Part D Subpart 2", official.Name == "PART D SUBPART 2")
        ]:
            print(name, official[filter].official_total_alloc.sum())

    # calculate coefficient of variation
    inputs = official.join(saipe_stacked.drop(columns="Name"), how="inner")\
        .astype({'Name': 'string'})

    if use_official_children:
        # replace SAIPE # children with official # children
        inputs["Estimated Population 5-17"] = inputs['official_5-17_pop.']

    return inputs


def _average_saipe(saipes):
    combined = pd.concat(saipes)
    # convert cv to variance
    combined["stderr"] = \
        (combined[
            'Estimated number of relevant children 5 to 17 years old '
            'in poverty who are related to the householder'
        ] * combined["cv"]).pow(2)
    agg = combined \
        .groupby(["State FIPS Code", "District ID"]) \
        .agg({
            'Name': 'first',
            'Estimated Total Population': 'mean',
            'Estimated Population 5-17': 'mean',
            'Estimated number of relevant children 5 to 17 years old '
            'in poverty who are related to the householder': 'mean',
            # variance of average of iid Gaussian is sum of variance over n^2
            'stderr': lambda stderr: np.sum(stderr) / (len(saipes)**2)
        })
    # convert variance back to cv
    agg['cv'] = np.sqrt(agg['stderr']) / agg["Estimated Total Population"]
    return agg.drop(columns='stderr')


def average_saipe(
    year: int, year_lag: int, verbose=True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get SAIPE averaged over `year_lag` years.

    Args:
        year (int): Most recent year.
        year_lag (int): Years to average over.
        verbose (bool, optional): Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Averaged SAIPE - district- and
            county-level.
    """
    if verbose:
        print(
            "Averaging SAIPEs:",
            [
                f"saipe{str(year)[2:]}"
                for year in range(year-year_lag, year+1)
            ]
        )
    combined = _average_saipe([
        get_saipe(f"{config.root}/data/saipe{str(year)[2:]}.xls")
        for year in range(year-year_lag, year+1)
    ])
    combined_county = _average_saipe([
        get_county_saipe(f"{config.root}/data/county_saipe{str(year)[2:]}.xls")
        for year in range(year-year_lag, year+1)
    ])
    return combined, combined_county


def get_acs_data(path: str, name: str):
    """Method for loading and formatting ACS data by school district from NCES
    [website](https://nces.ed.gov/programs/edge/TableViewer/acs/2018).

    Args:
        path (str): Path to txt file downloaded form the link above
        name (str): Sheet name.
    """
    data = pd.read_csv(path, sep="|", low_memory=False)
    # strip out NA district ID's
    # separate LEAID into FIPS code and district ID
    data["District ID"], data["State FIPS Code"] = \
        split_leaids(data.LEAID)
    data = data.set_index(["State FIPS Code", "District ID"])
    data = data.drop(
        columns=["GeoId", "Year", "Iteration"]
    )
    # drop 99999 - remainders
    data = data.query("`District ID` != 99999")

    varnames = pd.read_excel(
        os.path.join(
            config.root,
            "data/discrimination/ACS-ED_2015-2019_RecordLayouts.xlsx"
        ),
        sheet_name=name,
        index_col=0,
        engine='openpyxl'
    )

    drop = []
    new = {}
    for c in data.columns:
        try:
            newname = "{} ({}) - {}".format(
                varnames.loc[c].vlabel.split(";")[-1].lstrip(),
                varnames.loc[c].vlabel.split(";")[2].lstrip(),
                re.sub(r'\d+', '', c.split("_")[-1])
            )
            new[c] = newname if newname not in new.values() else c
        except KeyError:
            if c == "Geography":
                new[c] = c
            else:
                drop.append(c)

    data = data.drop(columns=drop).rename(columns=new)
    return data


def impute_missing(original, update, verbose=True):
    """Impute data for missing LEAs from another file.
    """
    for c in [c for c in original.columns if c not in update.columns]:
        update.loc[:, c] = np.nan
    update_reduced = update.loc[
        update.index.difference(original.index),
        original.columns
    ]
    imputed = pd.concat([
        original,
        update_reduced
    ])
    if verbose:
        print(
            "[INFO] Successfully imputed",
            len(update.index.difference(original.index)),
            "new indices"
        )
    return imputed


def get_acs_unified(verbose=False):
    """Load ACS demographic variables.
    """
    # get public school children data
    demographics_students = get_acs_data(
        f"{config.root}/data/discrimination/CDP05.txt",
        "CDP_ChildPop"
    )
    # update any missing with general pop data
    demographics = impute_missing(
        demographics_students,
        get_acs_data(
            f"{config.root}/data/discrimination/DP05.txt",
            "DP_TotalPop"
        )
    )
    social_students = get_acs_data(
        f"{config.root}/data/discrimination/CDP02.txt",
        "CDP_ChildPop"
    )
    social = impute_missing(
        social_students,
        get_acs_data(
            f"{config.root}/data/discrimination/DP02.txt",
            "DP_TotalPop"
        )
    )
    economic_students = get_acs_data(
        f"{config.root}/data/discrimination/CDP03.txt",
        "CDP_ChildPop"
    )
    economic = impute_missing(
        economic_students,
        get_acs_data(
            f"{config.root}/data/discrimination/DP03.txt",
            "DP_TotalPop"
        )
    )
    housing_students = get_acs_data(
        f"{config.root}/data/discrimination/CDP04.txt",
        "CDP_ChildPop"
    )
    housing = impute_missing(
        housing_students,
        get_acs_data(
            f"{config.root}/data/discrimination/DP04.txt",
            "DP_TotalPop"
        )
    )
    if verbose:
        print(demographics.shape)
        print(social.shape)
        print(economic.shape)
        print(housing.shape)
    return demographics\
        .join(social, lsuffix="_demo", rsuffix="_social", how="inner")\
        .join(economic, rsuffix="_econ", how="inner")\
        .join(housing, rsuffix="_housing", how="inner")


def split_leaids(leaids: pd.Series):
    # get the last seven digits of the ID
    leaids = leaids.astype(str).str[-7:]
    return leaids.str[-5:].astype(int), leaids.str[:-5].astype(int)


def get_sppe(path):
    fips_codes = pd.read_csv(f"{config.root}/data/fips_codes.csv").rename(
        columns={
            'FIPS': 'State FIPS Code'
        }
    )
    # quirk of original data file - need to change DC's name for join
    fips_codes.loc[fips_codes["Name"] == "District of Columbia", "Name"] = \
        "District Of Columbia Public Schools"
    sppe = pd.read_excel(path, header=2, engine='openpyxl')\
        .rename(columns={"Unnamed: 0": "Name"})[["Name", "ppe"]]
    return sppe.merge(fips_codes, on="Name", how="right")\
        .set_index("State FIPS Code")


def median_cv(total_pop: float) -> float:
    """
    Based on the table given here:
    https://www.census.gov/programs-surveys/saipe/guidance/district-estimates.html

    Args:
        total_pop (float): Total population of district.

    Returns:
        float: Coefficient of variation.
    """
    if total_pop <= 2500:
        return 0.67
    elif total_pop <= 5000:
        return 0.42
    elif total_pop <= 10000:
        return 0.35
    elif total_pop <= 20000:
        return 0.28
    elif total_pop <= 65000:
        return 0.23
    return 0.15


def weighting(eligible: float, pop: float) -> float:
    """Gradated weighting algorithm given in
    [Sonnenberg](https://nces.ed.gov/surveys/annualreports/pdf/titlei20160111.pdf).

    Returns weighted eligibility counts.

    Args:
        eligible (float): Number of eligible children.
        pop (float): Number total children.

    Returns:
        float: Weighted count.
    """
    # calculate weighted count based on counts
    wec_counts = 0
    for r, w in {
        (1, 691): 1.0,
        (692, 2262): 1.5,
        (2263, 7851): 2.0,
        (7852, 35514): 2.5,
        (35514, None): 3.0
    }.items():
        if r[1] is not None and eligible > r[1]:
            wec_counts += (r[1] - r[0] + 1) * w
        elif eligible >= r[0]:
            wec_counts += (eligible - r[0] + 1) * w

    # calculate weighted count based on proportions
    wec_props = 0
    for r, w in {
        (0, 0.1558): 1.0,
        (0.1558, 0.2211): 1.75,
        (0.2211, 0.3016): 2.5,
        (0.3016, 0.3824): 3.25,
        (0.3824, None): 4.0
    }.items():
        upper = floor(r[1]*pop) if r[1] is not None else None
        lower = ceil(r[0]*pop)

        if upper is not None and eligible > upper:
            wec_props += (upper - lower) * w
        elif eligible >= lower:
            wec_props += (eligible - lower) * w

    # take the higher weighted eligibility count
    return max(wec_counts, wec_props)


def data(
    inputs: pd.DataFrame,
    mechanism: Mechanism,
    sppe: pd.DataFrame,
    sampling_kwargs: dict = {},
    verbose: bool = True
) -> pd.DataFrame:
    """Prepare data needed to compute Title I allocations, including noised
    poverty estimates and other inputs.

    Args:
        inputs (pd.DataFrame): Ground truth poverty estimates.
        mechanism (Mechanism): Randomization mechanism.
        sppe (pd.DataFrame): State per-pupil expenditure.
        sampling_kwargs (dict, optional): Sampling parameters. Defaults to {}.
        verbose (bool, optional): Defaults to True.

    Returns:
        pd.DataFrame: Combined dataframe of inputs and noised estimates.
    """
    # ground truth - assume SAIPE 2019 is ground truth
    grants = inputs.rename(columns={
        "Estimated Total Population": "true_pop_total",
        "Estimated Population 5-17": "true_children_total",
        "Estimated number of relevant children 5 to 17 years old in poverty"
        " who are related to the householder": "true_children_poverty"
    })

    # sample from the sampling distribution
    mechanism_sampling = Sampled(**sampling_kwargs)
    grants["est_pop_total"], \
        grants["est_children_total"], \
        grants["est_children_poverty"] = mechanism_sampling.poverty_estimates(
            grants["true_pop_total"],
            grants["true_children_total"],
            grants["true_children_poverty"],
            grants["cv"]
        )

    # get the noise-infused estimates - after sampling
    grants["dpest_pop_total"], \
        grants["dpest_children_total"], \
        grants["dpest_children_poverty"] = mechanism.poverty_estimates(
        grants["est_pop_total"],
        grants["est_children_total"],
        grants["est_children_poverty"]
    )

    # back out the noise-infused estimates - before sampling
    # doing it this way because we want to see the same noise draws added to
    # both bases - not a separate draw here
    for var in ("pop_total", "children_total", "children_poverty"):
        grants[f"dp_{var}"] = \
            grants[f"dpest_{var}"] - grants[f"est_{var}"] \
            + grants[f"true_{var}"]

    # now, add in the number of foster, TANF, delinquent children
    # derived from final formula count reported by ESEA
    other_eligible = \
        inputs["official_total_formula_count"] \
        - grants["true_children_poverty"]

    for prefix in ("true", "est", "dp", "dpest"):
        grants[f"{prefix}_children_eligible"] = grants[
            f"{prefix}_children_poverty"
        ] + other_eligible

    # join in SPPE
    grants = grants.join(sppe["ppe"].rename('sppe'), how='left')

    if verbose and len(grants[grants.sppe.isna()]['Name'].values) > 0:
        print(
            "[WARN] Dropping districts with missing SPPE data:",
            grants[grants.sppe.isna()]['Name'].values
        )
    grants = grants.dropna(subset=["sppe"])
    grants.sppe = grants.sppe.astype(float)

    return grants
