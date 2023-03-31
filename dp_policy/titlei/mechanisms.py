import pandas as pd
from typing import Tuple
import numpy as np
from diffprivlib.mechanisms import Laplace as LaplaceMech
from diffprivlib.mechanisms import GaussianAnalytic as GaussianMech
from diffprivlib.accountant import BudgetAccountant
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class Mechanism:
    """
    A class for the different privacy mechanisms we employ to compute
    poverty estimates.
    """
    def __init__(
        self, sensitivity=2.0, round=False, clip=True, noise_total=False
    ):
        self.sensitivity = sensitivity
        self.round = round
        self.clip = clip
        self.noise_total = noise_total

    def poverty_estimates(
        self,
        pop_total: pd.Series,
        children_total: pd.Series,
        children_poverty: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns dataframe for children in poverty, children total, and total
        population indexed by district ID.

        Args:
            pop_total (pd.Series): Total population in each district.
            children_total (pd.Series): Total children in each district.
            children_poverty (pd.Series): Children in poverty in each district.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Noised versions
                of the input tuples.
        """
        raise NotImplementedError

    def post_processing(self, count: pd.Series) -> pd.Series:
        """Post processing methods for noised counts. (Rounding or clipping.)

        Args:
            count (pd.Series): Noised count to process.

        Returns:
            pd.Series: Processed count.
        """
        if self.round:
            count = np.round(count)
        if self.clip:
            count = np.clip(count, 0, None)
        return count


class GroundTruth(Mechanism):
    """No randomization.
    """
    def __init__(self, *args, **kwargs):
        pass

    def poverty_estimates(
        self, pop_total, children_total, children_poverty
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return pop_total, children_total, children_poverty


class DummyMechanism():
    """No randomization.
    """
    def randomise(self, x):
        return x


class DiffPriv(Mechanism):
    """Differentially private mechanisms wrapping `diffprivlib`.
    """
    def __init__(
        self, epsilon, delta, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.mechanism = None
        # for advanced composition
        self.accountant = BudgetAccountant(delta=self.delta)
        self.accountant.set_default()

    def poverty_estimates(
        self, pop_total, children_total, children_poverty
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if self.mechanism is None:
            raise NotImplementedError

        # NOTE: as of 3/21, by default only adding noise to poverty estimate
        # (for consistency with sampling, where est. var. is unavailable)
        children_poverty = children_poverty.apply(self.mechanism.randomise)
        if self.noise_total:
            children_total = children_total.apply(self.mechanism.randomise)

        # print("After estimation, privacy acc:", self.accountant.total())
        # no negative values, please
        # also rounding counts - post-processing
        return self.post_processing(pop_total),\
            self.post_processing(children_total),\
            self.post_processing(children_poverty)


class Laplace(DiffPriv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanism = LaplaceMech(
            epsilon=self.epsilon,
            delta=self.delta,
            sensitivity=self.sensitivity
        )


class Gaussian(DiffPriv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanism = GaussianMech(
            epsilon=self.epsilon,
            delta=self.delta,
            sensitivity=self.sensitivity
        )


class Sampled(Mechanism):
    """Mechanism for simulating sampling errors.
    """
    def __init__(
        self,
        *args,
        multiplier: float = 1.0,
        distribution: str = "gaussian",
        **kwargs
    ):
        """
        Args:
            multiplier (float, optional): Scales sampling noise by a constant.
                Defaults to 1.0.
            distribution (str, optional): Distribution of sampling noise.
                Supported options are 'gaussian' and 'laplace'. Defaults to
                "gaussian".
        """
        super().__init__(*args, **kwargs)
        # these are fixed, because sampling error
        # is theoretically immutable by algo means.
        # reported estimates are non-negative integers.
        self.clip = True
        self.round = True
        self.multiplier = multiplier
        self.distribution = distribution

    def poverty_estimates(
        self, pop_total, children_total, children_poverty, cv
    ):
        children_poverty = self._noise(children_poverty, cv,pop_total)
        if self.noise_total:
            # NOTE: assuming CVs are same for total children.
            # This is beyond Census guidance.
            children_total = self._noise(children_total, cv,pop_total)

        return self.post_processing(pop_total), \
            self.post_processing(children_total), \
            self.post_processing(children_poverty)

    def _noise(self, count: pd.Series, cv: pd.Series,pop_total) -> pd.Series:
        """Add sampling noise.

        Args:
            count (pd.Series): Count to add sampling noise to.
            cv (pd.Series): Coefficients of variation to use for sampling
                variance.

        Returns:
            pd.Series: Noised counts.
        """
        
        if self.distribution == "gaussian":
            noised = np.random.normal(
                count,  # mean
                count * cv * self.multiplier  # stderr
            )
            
        elif  self.distribution == "gaussian_laplace":
            count = count + np.random.laplace(0, 10, len(count))
            count[count<0] = 0
            noised = np.random.normal(
                count,  # mean
                np.sqrt((count * cv * self.multiplier)**2 + 200)   # stderr
            )   
            
        elif self.distribution == "gaussian_no_noise":
            noised = count
            
            
            
        elif self.distribution == "JS_no_noise":        
            count = count+1
            V = np.mean((count*cv)**2)
            k = len(count)
            r = 1
            
            beta_hat_0 = np.sum(count / V)  /(np.sum(1/V))
            B_hat_i = ((len(count)-2) *V/(sum(count**2)))
            noised = (1-B_hat_i)*count + B_hat_i * beta_hat_0 -1
            
            np.savetxt("JS/Avg_V.csv", np.array([V]), delimiter=',')
            np.savetxt("JS/beta_hat_0.csv", np.array([beta_hat_0]), delimiter=',')
            np.savetxt("JS/B_hat_i_JS.csv", np.array([B_hat_i]), delimiter=',')
            np.savetxt("JS/JS_shrunk_counts.csv", noised, delimiter=',')
            
            
            
            
        elif self.distribution == "JS_unequalvar_no_noise_H":
            count = count+1
            V = (count*cv)**2
            k = len(count)
            r = 1

            beta_hat_0 = np.sum(count / V)  /(np.sum(1/V))
            sigma_sq_hat = 1/(k-r) * np.sum((count - beta_hat_0)**2 /V )
            B_hat_H =   (k-r-2)/(k-4) * 1/sigma_sq_hat
            V_H = k/(np.sum(1/V))
            A_hat = V_H * (1-B_hat_H) /B_hat_H
            B_hat_i = (V / (V+ A_hat))
            noised = (1-B_hat_i)*count + B_hat_i * beta_hat_0 -1
            
            
            np.savetxt("H/count.csv", count-1, delimiter=",")
            np.savetxt("H/V.csv", V, delimiter=",")
            np.savetxt("H/beta_hat_0.csv", np.array([beta_hat_0]), delimiter=',')
            np.savetxt('H/sigma_sq_hat.csv',np.array([sigma_sq_hat]),delimiter=',')
            np.savetxt('H/V_H.csv',np.array([V_H]),delimiter=',')
            np.savetxt('H/A_hat.csv',np.array([A_hat]),delimiter=',')
            np.savetxt("H/B_hat_H.csv", np.array([B_hat_H]), delimiter=',')
            np.savetxt("H/B_hat_i.csv", B_hat_i, delimiter=',')
            np.savetxt("H/H_shrunk_counts.csv", noised, delimiter=',')
            
        elif self.distribution == "JS_unequalvar_no_noise_H_prop":        
            count = count+1
            np.savetxt("H_Prop/Count.csv", (count-1)*pop_total, delimiter=',')
            
            V = (count*cv)**2
            k = len(count)
            r = 1
            
            count = count/(pop_total+1)
            V = V/((pop_total+1)**2)
            
            beta_hat_0 = np.sum(count / V)  /(np.sum(1/V))            
            sigma_sq_hat = 1/(k-r) * np.sum((count - beta_hat_0)**2 /V )
            B_hat_H =   (k-r-2)/(k-4) * 1/sigma_sq_hat
            V_H = k/(np.sum(1/V))
            A_hat = V_H * (1-B_hat_H) /B_hat_H
            B_hat_i = (V / (V+ A_hat))
            noised = (1-B_hat_i)*count + B_hat_i * beta_hat_0 
            noised[noised<=0] = 0
            
            
            np.savetxt("H_Prop/V.csv", V, delimiter=',')
            np.savetxt("H_Prop/Prop.csv", count, delimiter=',')
            np.savetxt("H_Prop/Shrunk_Prop.csv", noised, delimiter=',')
            noised = (noised * (pop_total+1))-1
            noised[noised<=0] = 0
            
            
            np.savetxt("H_Prop/pop_total.csv", pop_total, delimiter=',')
            
            np.savetxt("H_Prop/beta_hat_0_Prop.csv", np.array([beta_hat_0]), delimiter=',')
            np.savetxt("H_Prop/B_hat_i_H_Prop.csv", B_hat_i, delimiter=',')
            np.savetxt("H_Prop/JS_shrunk_counts_Prop.csv", noised, delimiter=',')
     
        
            

        elif self.distribution == "JS_unequalvar_no_noise_HB":
            count = count+1
            V = (count*cv)**2
            k = len(count)
            r = 1

            beta_hat_0 = np.sum(count / V)  /(np.sum(1/V))
            B_hat_i = ((k-2)/V)/(np.sum((count / V)**2))
            B_hat_i[B_hat_i>1] = 1
            noised = (1-B_hat_i)*count + B_hat_i * beta_hat_0 -1
            
            np.savetxt("HB/count.csv", count-1, delimiter=",")
            np.savetxt('HB/V.csv',V,delimiter=',')
            np.savetxt("HB/B_hat_i_HB.csv", B_hat_i, delimiter=',')
            np.savetxt('HB/beta_hat_0.csv',np.array([beta_hat_0]),delimiter = ',')
            np.savetxt("HB/HB_shrunk_count.csv", noised, delimiter=',')
            
            
        elif self.distribution == "JS_unequalvar_no_noise_H_reg":
            count = count+1
            V = (count*cv)**2
            k = len(count)
            r = 2

            X = np.array(pop_total).transpose().reshape(-1, 1)
            ones = np.ones([X.shape[0],X.shape[1]])
            X = np.append(np.ones([len(pop_total),1]),X,1)      
            reg = LinearRegression(fit_intercept= False).fit(X, count,sample_weight = 1/V)
            beta_hat_0 = reg.predict(X)
            
            sigma_sq_hat = 1/(k-r) * np.sum((count - beta_hat_0)**2 /V )
            B_hat_H =   (k-r-2)/(k-4) * 1/sigma_sq_hat
            V_H = k/(np.sum(1/V))
            A_hat = V_H * (1-B_hat_H) /B_hat_H
            B_hat_i = (V / (V+ A_hat))
            noised =  (1-B_hat_i)*count + B_hat_i * beta_hat_0 -1   
             
            
            
            
            
            
        elif self.distribution == "JSGaussian":
            V = np.mean((count*self.multiplier* cv)**2)
            shrinkage_factor = 1-((len(count)-2) *V/(sum(count**2)))
            mu_js = shrinkage_factor*count
            
            noised = np.random.normal(
                mu_js ,  # mean
                np.sqrt(shrinkage_factor **2   * V)
   
            )

            
        elif self.distribution == "JSGaussian_unequalvar_H":
            count = count+1
            V = (count*cv)**2
            k = len(count)
            r = 1

            beta_hat_0 = np.sum(count / V)  /(np.sum(1/V))
            sigma_sq_hat = 1/(k-r) * np.sum((count - beta_hat_0)**2 /V )
            B_hat_H =   (k-r-2)/(k-4) * 1/sigma_sq_hat
            V_H = k/(np.sum(1/V))
            A_hat = V_H * (1-B_hat_H) /B_hat_H
            B_hat_i = (V / (V+ A_hat))
            
            noised = np.random.normal(
                (1-B_hat_i)*count + B_hat_i * beta_hat_0 -1     ,  # mean
                 np.sqrt( V *(1-B_hat_i) )  # stderr
            )
            
        elif self.distribution == "JSGaussian_unequalvar_H_prop":        
            count = count+1
            V = (count*cv)**2
            k = len(count)
            r = 1
            #print(count)
            #print(pop_total)
            
            count = count/(pop_total+1)
            V = V/(pop_total**2)
            #print(count)
            
            beta_hat_0 = np.sum(count / V)  /(np.sum(1/V))
            sigma_sq_hat = 1/(k-r) * np.sum((count - beta_hat_0)**2 /V )
            B_hat_H =   (k-r-2)/(k-4) * 1/sigma_sq_hat
            V_H = k/(np.sum(1/V))
            A_hat = V_H * (1-B_hat_H) /B_hat_H
            B_hat_i = (V / (V+ A_hat))
            
            #print((1-B_hat_i)*count)
            #print(beta_hat_0)
            noised = (pop_total+1)*np.random.normal(
                (1-B_hat_i)*count + B_hat_i * beta_hat_0     ,  # mean
                 np.sqrt( V *(1-B_hat_i) )  # stderr
            )
            noised[noised<=0] = 0
            #print(noised)
            
        elif self.distribution == "JSGaussian_unequalvar_H_prop_laplace": 
            #print(count)
            #print('Proportion wo/lap')
            orignal_prop = count/(pop_total+1)
            #print(orignal_prop)
            
            count = count+1
            lap_noise = np.random.laplace(0, 10, len(count))
            count = count + lap_noise
        
            
            
            
            count[count<=0] = 0
            count[count>pop_total] = pop_total
            
            #pop_total = pop_total + lap_noise
            #pop_total[pop_total<=0] = 1
            #print(pop_total)
            
            
            V = ((count-lap_noise)*cv)**2
            k = len(count)
            r = 1
            
            #count = count/(pop_total+lap_noise+1)
            count = count/(pop_total+1)
            count[count<=0] = 0
            count[count>=1] = 1
            
            #print('Proportion wo/lap')
            #print(count)
            
            #plt.plot(orignal_prop, count, 'ro')
            #plt.show()
            
            
            
            V = V/(pop_total**2)
            
            beta_hat_0 = np.sum(count / V)  /(np.sum(1/V))
            sigma_sq_hat = 1/(k-r) * np.sum((count - beta_hat_0)**2 /V )
            B_hat_H =   (k-r-2)/(k-4) * 1/sigma_sq_hat
            V_H = k/(np.sum(1/V))
            A_hat = V_H * (1-B_hat_H) /B_hat_H
            B_hat_i = (V / (V+ A_hat))
            
            
            #print((1-B_hat_i)*count)
            #print(beta_hat_0)
            noised = (pop_total+1)*np.random.normal(
                (1-B_hat_i)*count + B_hat_i * beta_hat_0     ,  # mean
                 np.sqrt( V *(1-B_hat_i) + 200/(pop_total**2) )  # stderr
            )
            noised[noised<=0] = 1
            #idx = noised>=pop_total
            #noised[idx] = pop_total[idx]
            #print(noised)
            

        elif self.distribution == "JSGaussian_unequalvar_HB":
            count = count+1
            V = (count*cv)**2
            k = len(count)
            r = 1

            beta_hat_0 = np.sum(count / V)  /(np.sum(1/V))
            B_hat_i = ((k-2)/V)/(np.sum((count / V)**2))
            B_hat_i[B_hat_i>=1] = 1
            B_hat_i[B_hat_i<=0] = 0
            
            noised = np.random.normal(
                (1-B_hat_i)*count + B_hat_i * beta_hat_0 -1     ,  # mean
                 np.sqrt( V *(1-B_hat_i) )  # stderr
            )            
            
            
            
        elif self.distribution == "JSGaussian_unequalvar_HB_lapalce":
            count = count+1
            count = count + np.random.laplace(0, 10, len(count))
            count[count<0] = 0
            
            V = (count*cv)**2
            k = len(count)
            r = 1

            beta_hat_0 = np.sum(count / V)  /(np.sum(1/V))
            B_hat_i = ((k-2)/V)/(np.sum((count / V)**2))
            B_hat_i[B_hat_i>=1] = 1
            B_hat_i[B_hat_i<=0] = 0
            
            noised = np.random.normal(
                (1-B_hat_i)*count + B_hat_i * beta_hat_0 -1     ,  # mean
                 np.sqrt( V *(1-B_hat_i) +200 )  # stderr
            )   
            
            
        elif self.distribution == "JSGaussian_unequalvar_H_reg":
            count = count+1
            V = (count*cv)**2
            k = len(count)
            r = 2

            X = np.array(pop_total).transpose().reshape(-1, 1)
            ones = np.ones([X.shape[0],X.shape[1]])
            X = np.append(np.ones([len(pop_total),1]),X,1)      
            reg = LinearRegression(fit_intercept= False).fit(X, count,sample_weight = 1/V)
            beta_hat_0 = reg.predict(X)
            
            sigma_sq_hat = 1/(k-r) * np.sum((count - beta_hat_0)**2 /V )
            B_hat_H =   (k-r-2)/(k-4) * 1/sigma_sq_hat
            V_H = k/(np.sum(1/V))
            A_hat = V_H * (1-B_hat_H) /B_hat_H
            B_hat_i = (V / (V+ A_hat))
            noised = np.random.normal(
                (1-B_hat_i)*count + B_hat_i * beta_hat_0 -1     ,  # mean
                 np.sqrt( V *(1-B_hat_i) )  # stderr
            )      
                
            
            
        elif self.distribution == "laplace":
            noised = np.random.laplace(
                # mean
                count,
                # variance is 2b^2 = (count * cv)^2
                # b = count * cv * sqrt(1/2)
                np.sqrt(0.5) * count * cv * self.multiplier
            )
        else:
            raise ValueError(
                f"{self.distribution} is not a valid distribution."
            )
        return np.clip(
            noised,
            0,
            None
        )
