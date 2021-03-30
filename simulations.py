import numpy as np
import pandas as pd
import scipy.stats as sp
from itertools import compress
import statsmodels.discrete.discrete_model as sm
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'genplotpy'))
import torch
from genplotpy import models
from scipy.stats import gaussian_kde
# need to add method to check independent sims better


class BaseSim:
    """
    Base simulation class with LD methods for use with all SNPs
    """
    def __init__(self, seed=1):
        self.seed = seed
        self.block_sizes = None
        self.ld = None
        self.col_sample_idx = None

    def set_seed(self, seed):
        self.seed = seed

    def add_simple_ld(self, causal_snp, ld_type, verbose=False, reshape=False):
        # returns numpy array of snp in LD with causal SNP
        if isinstance(causal_snp, pd.Series):
            causal_snp = causal_snp.values
        assert isinstance(ld_type, float)
        assert isinstance(causal_snp, np.ndarray)

        indices_to_shuffle = np.random.choice(np.arange(causal_snp.shape[0]),
                                              int(round(1-ld_type, 2) * causal_snp.shape[0]), replace=False)
        ld_snp = np.copy(causal_snp)
        ld_snp[indices_to_shuffle] = np.random.permutation(causal_snp[indices_to_shuffle])

        if verbose:
            print('Observed correlation: {}, Expected correlation: {}'.format(self.check_ld(causal_snp, ld_snp),
                                                                              round(ld_type, 2)))
        if reshape:
            ld_snp = ld_snp.reshape(ld_snp.shape[0], 1)

        return ld_snp

    def replace_with_2_snp_ld(self, X, idx_to_swap_for_ld, ld, y=None, cols=None):
        ld_snps = np.hstack(
            ([self.add_simple_ld(X.iloc[:, idx], x, reshape=True) for idx, x in zip(idx_to_swap_for_ld, ld)]))
        if y is not None:
            df_ld = pd.DataFrame(np.hstack([y.values.reshape(-1, 1), ld_snps]))
        else:
            df_ld = pd.DataFrame(ld_snps)
        if cols is not None:
            df_ld.columns = cols

        return df_ld

    def add_simple_ld_block(self, causal_snp, ld=None, keep_causal=True, block_size=10, make_symmetric=False,
                            sort_ld=False, ld_type='uniform', seed=None):
        if seed is not None:
            np.random.seed(seed)
        # final block size is (2 * block_size) + 1 if make_symmetric is True
        ld_mid = 0
        if ld is None:
            if ld_type == 'uniform':
                ld = np.random.random_sample(block_size)
            elif isinstance(ld_type, (float, int)):
                assert 0 <= ld_type <= 1, 'Pass ld_type between 0 and 1'
                ld = np.repeat(ld_type, block_size)
            else:
                raise ValueError('ld_type not supported')
            if sort_ld:
                ld = np.sort(ld)
            if make_symmetric:
                if keep_causal:
                    ld_mid = ld.shape[0]
                    ld = np.hstack((ld, 1.0, np.flip(ld)))
                else:
                    ld = np.hstack((ld, np.flip(ld)))
            else:
                if keep_causal:
                    ld_mid = int(ld.shape[0]/2)
                    ld = np.hstack((ld[:ld_mid], 1.0, ld[ld_mid:]))

        self.ld = ld
        ld_block = np.hstack(([self.add_simple_ld(causal_snp, x, reshape=True) for x in ld]))

        return ld_block, ld_mid

    def replace_with_ld_block(self, causal_snps, block_sizes='varied', p_limit=None, ld=None, keep_causal=True,
                              make_symmetric=False, sort_ld=False, ld_type='uniform', seed=None):
        if seed is not None:
            np.random.seed(self.seed)
        # , indices_to_replace=None
        # currently replaces all, but should ultimately take a list of SNPs to replace with blocks
        if isinstance(block_sizes, str):
            assert block_sizes == 'varied'
            block_sizes = np.random.randint(1, 11, causal_snps.shape[1])
        elif isinstance(block_sizes, list):
            block_sizes = np.array(block_sizes)
        elif isinstance(block_sizes, int):
            block_sizes = np.repeat(block_sizes, causal_snps.shape[1])

        assert block_sizes.shape[0] == causal_snps.shape[1]
        self.block_sizes = block_sizes

        new_snps, causal_idx_list = [], []
        size_counter = -1
        for x in np.arange(block_sizes.shape[0]):
            block, causal_idx = self.add_simple_ld_block(causal_snps.iloc[:, x].values, block_size=block_sizes[x],
                                                         keep_causal=keep_causal, make_symmetric=make_symmetric,
                                                         ld=ld, sort_ld=sort_ld, ld_type=ld_type, seed=seed)
            causal_idx_list.append(size_counter + causal_idx + 1)
            new_snps.append(block)
            size_counter += block.shape[1]

        # sample LD blocks down to p_limit, but keep causal snps if present
        if seed is not None:
            np.random.seed(self.seed)  # reset seed as ld shuffling above affected by n, so changes values here
        causal_idx_list = np.array(causal_idx_list)
        new_snps = np.hstack(new_snps)
        if all([isinstance(p_limit, int), keep_causal]):
            if p_limit < new_snps.shape[1]:
                # downsample LD SNPs
                self.col_sample_idx = self.sample_ld_to_p(p_limit, np.arange(new_snps.shape[1]), causal_idx_list)
                new_snps = new_snps[:, self.col_sample_idx]
            elif p_limit > new_snps.shape[1]:
                # add random SNPs
                self.col_sample_idx = np.arange(new_snps.shape[1])
                noise = self.add_noise(new_snps.shape[0], p_limit - new_snps.shape[1])
                new_snps = np.hstack([new_snps, noise])
        elif all([isinstance(p_limit, int), not keep_causal]):
            if p_limit < new_snps.shape[1]:
                # downsample all SNPs as no causal snps
                self.col_sample_idx = np.sort(
                        np.random.choice(np.arange(new_snps.shape[1]), size=p_limit, replace=False))
                new_snps = new_snps[:, self.col_sample_idx]
            elif p_limit > new_snps.shape[1]:
                # add random SNPs
                self.col_sample_idx = np.arange(new_snps.shape[1])
                noise = self.add_noise(new_snps.shape[0], p_limit - new_snps.shape[1])
                new_snps = np.hstack([new_snps, noise])

        return pd.DataFrame(new_snps)

    def yhat(self, clf, X):
        if isinstance(clf, torch.nn.Module):
            with torch.no_grad():
                return self.coerce_to_numpy(torch.sigmoid(clf(self.coerce_to_torch(X)))[:, 1])
        else:
            if hasattr(clf, 'predict_proba'):
                return clf.predict_proba(X)[:, 1]
            elif hasattr(clf, 'decision_function'):
                return clf.decision_function(X)
            elif hasattr(clf, 'predict'):
                return clf.predict(X)
            else:
                raise ValueError('clf has no predict attr')

    @staticmethod
    def plot_ld(df, figsize=(11, 9), ax=None, annotate=False, palette='Blues'):
        corrr = df.corr()
        sns.set(style="white")
        mask = np.zeros_like(corrr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Generate a custom diverging colormap
        # cmap = sns.diverging_palette(220, 10, as_cmap=True)
        cmap = sns.color_palette(palette)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corrr, mask=mask, cmap=cmap, annot=annotate, vmin=0, vmax=1, square=True, linewidths=.5,
                    cbar_kws={"shrink": .35}, ax=ax, yticklabels=False, xticklabels=False)

        return ax

    @staticmethod
    def sample_ld_to_p(p_lim, x, c):
        mask = np.ones(x.size, dtype=bool)
        mask[c] = False
        result = x[mask]

        sample_to = p_lim - c.shape[0]
        keep_idx = np.sort(np.hstack([np.random.choice(result, size=sample_to, replace=False), c]))

        return keep_idx

    @staticmethod
    def add_noise(n, p, maf=(0.05, 0.5)):
        maf = np.random.uniform(maf[0], maf[1], p)
        genotypes = np.array([np.random.binomial(2, freq, n) for freq in maf]).T

        return genotypes

    @staticmethod
    def check_ld(snp1, snp2):

        return round(np.corrcoef(snp1, snp2)[0, 1], 2)

    @staticmethod
    def check_betas(data):
        empirical_betas = []
        for i in range(data.shape[1] - 1):
            model = sm.Logit(data.iloc[:, 0], sm.tools.add_constant(data.iloc[:, i + 1]))
            result = model.fit(disp=0)
            empirical_betas.append(result.params[1])

        return np.array(empirical_betas)

    @staticmethod
    def sample_pardinas_odds_ratios(n):
        sim_data = pd.read_csv(os.path.join(os.path.expanduser('~'), 'gensimpy/data/pardinas_supp_table_3.csv'))
        or_kde = gaussian_kde(sim_data['OR'], bw_method=0.1)

        return np.squeeze(or_kde.resample(n))

    @staticmethod
    def coerce_to_numpy(X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        if isinstance(X, list):
            X = np.array(X)
        assert isinstance(X, np.ndarray), 'Supply X as numpy array or Torch tensor'

        return X

    @staticmethod
    def coerce_to_torch(X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).type('torch.FloatTensor')
        assert isinstance(X, torch.Tensor), 'Supply X as numpy array or Torch tensor'

        return X

    @staticmethod
    def islice_col(X, col):
        if hasattr(X, 'iloc'):
            return X.iloc[:, col]
        else:
            return X[:, col]


class IndependentSim(BaseSim):
    """
    Collection of methods for simulating SNPs independently under a dominant model
    """
    def __init__(self, n_snps=100, n_samples=1000, case_control_ratio=0.5, prop=0.8, effects='range',
                 odds_ratio_choices=None, case_maf_choices=None, control_maf_choices=None, seed=1):
        super().__init__(seed)
        self.n_snps = n_snps
        self.n_samples = n_samples
        self.case_control_ratio = case_control_ratio
        self.effect_sizes = effects
        self.seed = seed

        self.proportion_associated = prop
        self.n_cases = int(self.case_control_ratio * self.n_samples)
        self.n_controls = int(self.n_samples - self.n_cases)
        self.n_associated = int(self.proportion_associated * self.n_snps)
        self.odds_ratio_choices = odds_ratio_choices
        self.case_maf_choices = case_maf_choices
        self.control_maf_choices = control_maf_choices
        self.causal_idx = None

    def set_parameters(self):
        print('Current seed: {}'.format(self.seed))
        np.random.seed(self.seed)

        if isinstance(self.effect_sizes, (int, float)):
            self.effect_sizes = float(self.effect_sizes)
            self.odds_ratio_choices = np.repeat(self.effect_sizes, self.n_associated)
        elif isinstance(self.effect_sizes, str):
            if self.effect_sizes == 'range':
                self.odds_ratio_choices = np.exp(np.random.normal(loc=0, scale=0.1, size=self.n_associated))
            elif self.effect_sizes == 'pardinas':
                self.odds_ratio_choices = self.sample_pardinas_odds_ratios(self.n_associated)
            else:
                raise ValueError('Only string option "range" recognised')
        else:
            raise TypeError('Must be float or str')
        self.causal_idx = np.arange(self.odds_ratio_choices.shape[0])
        self.odds_ratio_choices = np.hstack((self.odds_ratio_choices,
                                             np.repeat(1.0, self.n_snps - self.n_associated)))
        # np.random.shuffle(self.odds_ratio_choices)
        self.case_maf_choices = np.random.uniform(low=0.05, high=0.5, size=self.n_snps)

    def simulate(self, keep_sim_params=False, n_case=None, n_control=None, seed=None):
        self.seed = seed if seed is not None else self.seed
        if not keep_sim_params:
            self.set_parameters()
            self.control_maf_choices = (self.case_maf_choices /
                                        (self.odds_ratio_choices * (1 - self.case_maf_choices) + self.case_maf_choices))
        self.n_cases = n_case if n_case is not None else self.n_cases
        self.n_controls = n_control if n_control is not None else self.n_controls
        self.n_samples = int(n_case + n_control) if all([x is not None for x in (n_case, n_control)]) else self.n_samples

        case_control_status = (np.hstack((np.ones(self.n_cases), np.zeros(self.n_controls)))
                                 .astype(int)
                                 .reshape(self.n_samples, 1))

        # generate the genotypes
        genotypes = np.vstack(
            (np.array([np.random.binomial(2, maf, self.n_cases) for maf in self.case_maf_choices]).T,
             np.array([np.random.binomial(2, maf, self.n_controls) for maf in self.control_maf_choices]).T))
        genotypes = np.hstack((case_control_status, genotypes))

        self.seed += 1

        data = pd.DataFrame(genotypes)
        data.columns = ['Status'] + ['SNP_{}'.format(x + 1) for x in range(self.n_snps)]

        return data


class LiabilitySim(BaseSim):
    """
    Collection of methods for jointly simulating SNPs under an additive model
    """
    def __init__(self, n_snps=100, n_samples=500, proportion_causal=1.0, h2_liability=0.2, prevalence=0.01,
                 proportion_cases=0.5, n_population=250000, method='undersample_controls',
                 non_causal='random', ld_params=None, seed=1):
        super().__init__(seed)
        self.n_snps = n_snps
        self.n_samples = n_samples
        self.proportion_causal = proportion_causal
        self.n_associated = int(self.proportion_causal * self.n_snps)
        self.heritability_liability = h2_liability
        self.n_population = n_population
        self.prevalence = prevalence
        self.proportion_cases = proportion_cases
        self.method = method
        self.seed = seed
        self.non_causal = non_causal
        self.ld_params = ld_params
        self.maf = None
        self.effect_sizes = None
        self.error = None
        self.threshold = None
        self.causal_idx = None
        self.betas_for_prs = None
        self.phenotypic_value = None
        self.phenotypic_value_std = None
        self.genotypic_value = None
        self.empirical_heritability = None
        self.empirical_betas = None
        self.beta_correlation = None
        self.varg = None
        self.varp = None
        self.vare = None

    def simulate(self, keep_sim_params=False, set_pop=None, set_sample=None, set_method=None, verbose=True, seed=None):
        self.n_population = set_pop if set_pop is not None else self.n_population
        self.n_samples = set_sample if set_sample is not None else self.n_samples
        self.method = set_method if set_method is not None else self.method
        self.seed = seed if seed is not None else self.seed
        if verbose:
            print('--> Starting simulation with seed: {}'.format(self.seed))
            if keep_sim_params:
                print('Keeping MAF, betas and causal SNP index from last simulation')
        np.random.seed(self.seed)
        if not keep_sim_params:
            self.maf = np.random.uniform(0.05, 0.5, self.n_snps)
            self.effect_sizes = np.random.normal(0, 1, self.n_associated)

        if self.method == 'undersample_controls':
            n_cases = int(self.n_samples * self.proportion_cases)
        elif self.method == 'oversample_cases':
            n_cases = int(self.n_population * self.proportion_cases)
            # self.n_population = self.n_samples
        elif self.method == 'None':
            pass
        else:
            raise ValueError('method not recognised')

        genotypes = np.array([np.random.binomial(2, freq, self.n_population) for freq in self.maf]).T
        if not keep_sim_params:
            self.causal_idx = np.random.choice(np.arange(genotypes.shape[1]), replace=False, size=self.n_associated)

        causal_genotypes = np.copy(genotypes[:, self.causal_idx])
        self.genotypic_value = np.dot(causal_genotypes, self.effect_sizes.reshape(-1, 1))
        self.varg = self.genotypic_value.var()

        simulated_error_variance = (self.varg / self.heritability_liability) - self.varg
        self.error = np.random.normal(0, np.sqrt(simulated_error_variance),
                                      self.n_population).reshape(self.n_population, 1)
        self.vare = self.error.var()
        self.phenotypic_value = self.genotypic_value + self.error
        self.varp = self.phenotypic_value.var()
        self.empirical_heritability = round(self.varg / self.varp, 2)
        if verbose:
            print('Var(g): {0:.3f}, Var(e): {1:.3f}, Var(p): {2:.3f}'.format(self.varg, self.vare, self.varp))
            print('Simulated h2 on liability scale: {0}, Empirical h2 on liability scale: {1:.3f}'.format(
                  self.heritability_liability, self.empirical_heritability))

        self.threshold = sp.norm.ppf(1 - self.prevalence)

        self.phenotypic_value_std = (self.phenotypic_value - self.phenotypic_value.mean()) / self.phenotypic_value.std()
        binary_phenotype = np.where(self.phenotypic_value_std > self.threshold, 1, 0)

        df = pd.DataFrame(genotypes)
        df.columns = ['SNP_{}'.format(x + 1) for x in range(self.n_snps)]
        df['Status'] = binary_phenotype
        cols = df.columns.tolist()
        cols.insert(0, cols.pop())
        df = df.loc[:, cols]

        if self.method == 'undersample_controls':
            df = (df.query('Status == 1')
                    .sample(n=n_cases, replace=False)
                    .append(df.query('Status == 0')
                              .sample(n=self.n_samples-n_cases, replace=False)))
        elif self.method == 'oversample_cases':
            df = (df.query('Status == 1')
                    .sample(n=n_cases, replace=True)
                    .append(df.query('Status == 0')
                              .sample(n=self.n_population-n_cases, replace=False)))
        elif self.method == 'None':
            return df
        else:
            raise ValueError('method not recognised')

        df = df.reset_index(drop=True)
        self.seed += 1
        self.empirical_betas = self.check_betas(df.iloc[:, [0] + (self.causal_idx + 1).tolist()])
        self.beta_correlation = np.corrcoef(self.effect_sizes, self.empirical_betas)[0, 1]
        self.betas_for_prs = np.zeros(self.n_snps)
        self.betas_for_prs[self.causal_idx] = self.effect_sizes
        if verbose:
            print('Correlation between expected and actual betas: {0:.3f}'.format(self.beta_correlation))
            print('{} cases and {} controls simulated\n'.format(
                df[df.Status == 1].shape[0], df[df.Status == 0].shape[0]))

        return df


class InteractionSim(BaseSim):
    """
    Simulation of 2-SNP interaction models
    !! Add X-squared test to check_sim for observed and expected genotype counts
    !! Poossibly add probability of observing genotypes in resamples
    !! Add observed and expected maf and whether it is under HWE or not
    """
    def __init__(self, theta, alpha=1, seed=1):
        super().__init__(seed)
        self.alpha = alpha
        self.theta = theta
        self.seed = seed
        self.model_name = None
        self.model = None
        self.maf_a = None
        self.maf_b = None
        self.hwe_a = None
        self.hwe_b = None
        self.n_cases = None
        self.n_controls = None
        self.pgi_controls = None
        self.pgi_cases = None
        self.X = None
        self.y = None

    def set_seed(self, seed):
        self.seed = seed

    def set_model(self, model_name):
        model = models.TwoSnpModels(self.alpha, self.theta)

        self.model_name = model_name
        self.model = model.lookup_model(model_name)

    def set_maf(self, maf_a, maf_b):
        self.maf_a = maf_a
        self.maf_b = maf_b
        self.set_hwe_locus_a()
        self.set_hwe_locus_b()

    def set_hwe_locus_a(self):
        p = self.maf_a
        q = 1 - p
        self.hwe_a = np.hstack((p ** 2, 2 * p * q, q ** 2))

    def set_hwe_locus_b(self):
        p = self.maf_b
        q = 1 - p
        self.hwe_b = np.hstack((p ** 2, 2 * p * q, q ** 2))

    def control_pgi(self):
        # creates P(Gi), the vector of probabilities of each two-locus genotype combination
        self.pgi_controls = np.hstack((self.hwe_a[0] * self.hwe_b,
                                       self.hwe_a[1] * self.hwe_b,
                                       self.hwe_a[2] * self.hwe_b))

    def case_pgi(self):
        self.pgi_cases = np.multiply(self.pgi_controls, self.model)

    def set_parameters(self, n_cases, n_controls, maf_a, maf_b, model):
        self.n_cases = n_cases
        self.n_controls = n_controls
        self.set_model(model)
        self.set_maf(maf_a, maf_b)
        self.control_pgi()
        self.case_pgi()

    def simulate(self, n_cases=None, n_controls=None, seed=None):
        self.n_cases = n_cases if n_cases is not None else self.n_cases
        self.n_controls = n_controls if n_controls is not None else self.n_controls
        self.seed = seed if seed is not None else self.seed
        print('Current seed: {}'.format(self.seed))
        np.random.seed(self.seed)

        alleles = np.array(["aa_bb", "aa_Bb", "aa_BB", "Aa_bb", "Aa_Bb", "Aa_BB", "AA_bb", "AA_Bb", "AA_BB"])
        interactions = np.hstack((self._weighted_random_choice(alleles, self.n_cases, self.pgi_cases),
                                  self._weighted_random_choice(alleles, self.n_controls, self.pgi_controls)))

        snps = (pd.DataFrame({'snps': interactions})
                  .snps.str.split('_', expand=True)
                  .replace({'aa': 2, 'aA': 1, 'Aa': 1, 'AA': 0,
                            'bb': 2, 'bB': 1, 'Bb': 1, 'BB': 0})
                  .rename(columns={0: 'SNP_A', 1: 'SNP_B'}))
        snps['Status'] = np.hstack((np.repeat(1, self.n_cases), np.repeat(0, self.n_controls)))
        self.set_seed(self.seed + 1)
        self.X = snps.loc[:, ['SNP_A', 'SNP_B']]
        self.y = snps.loc[:, 'Status']

        return snps.loc[:, ['Status', 'SNP_A', 'SNP_B']]

    def check_sim(self):
        bar = '####################################'
        print('{} {} {}'.format(bar, 'Genotype Proportions', bar))
        print('SNP_A\n'
              'Expected Genotype Proportions: {}\n'
              'Empirical Genotype Proportions: {}\n'.format(self.hwe_a,
                                                            self.X.SNP_A.value_counts(normalize=True)
                                                            .loc[[2, 1, 0]].values.tolist()))
        print('SNP_B\n'
              'Expected Genotype Proportions: {}\n'
              'Empirical Genotype Proportions: {}\n'.format(self.hwe_b,
                                                            self.X.SNP_B.value_counts(normalize=True)
                                                            .loc[[2, 1, 0]].values.tolist()))

        observed_counts_cases = self.X[self.y == 1].apply(lambda x: ''.join(x.astype(str).tolist()),
                                                          axis=1).value_counts()
        observed_counts_controls = self.X[self.y == 0].apply(lambda x: ''.join(x.astype(str).tolist()),
                                                             axis=1).value_counts()
        observed_counts_cases = self._append_missing_counts(observed_counts_cases)
        observed_counts_controls = self._append_missing_counts(observed_counts_controls)

        # show raw sample counts in cases and controls
        print('{} {} {}'.format(bar, 'Two-Locus Genotype Counts', bar))
        genotypes = ['00', '10', '01', '11', '02', '20', '12', '21', '22']
        print('Observed Counts in Cases')
        case_counts = [observed_counts_cases[x] for x in genotypes]
        self.pprint_vector([case_counts[i] for i in [4, 6, 8, 2, 3, 7, 0, 1, 5]])
        print('\nObserved Counts in Controls')
        control_counts = [observed_counts_controls[x] for x in genotypes]
        self.pprint_vector([control_counts[i] for i in [4, 6, 8, 2, 3, 7, 0, 1, 5]])

        # calculate odds and odds ratios (where AABB is baseline risk)
        print('\n{} {} {}'.format(bar, 'Two-Locus Odds Ratios', bar))
        odds = [observed_counts_cases[x] / observed_counts_controls[x] for x in genotypes]
        odds_ratios = [x / odds[0] for x in odds]
        print('Observed OR w.r.t baseline aabb')
        self.pprint_vector(['%.2f' % odds_ratios[i] for i in [4, 6, 8, 2, 3, 7, 0, 1, 5]])
        print('\nExpected OR')
        self.pprint_vector([self.model[x] for x in [2, 1, 0, 5, 4, 3, 8, 7, 6]])

    def check_sim_boundary(self, clf=None, auc_pos=(2.1, 2.1), auc_size=15):
        clf = SVC(kernel="rbf", probability=False).fit(self.X, self.y) if clf is None else clf
        model_type = self.scatter_model(self.model_name)
        score = roc_auc_score(self.y, self.yhat(clf, self.X))

        ax = self.plot_decision_boundary(self.X, clf, score=score, ax=None, scatter_model=model_type,
                                         auc_pos=auc_pos, auc_size=auc_size)

        return ax

    def plot_decision_boundary(self, X, clf, score=None, ax=None, scatter_model=None, auc_pos=(2.1, 2.1), auc_size=15,
                               drop_tick_labels=False):
        # setup
        cm = plt.cm.RdYlGn
        cm_bright = plt.cm.Blues
        scatter_kwargs = dict(cmap=cm_bright, edgecolors='black', s=70, linewidths=1.5)
        h = .02  # step size in the mesh
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.subplot(1, 1, 1)

        # meshgrid
        x_min, x_max = self.islice_col(X, 0).min() - .5, self.islice_col(X, 0).max() + .5
        y_min, y_max = self.islice_col(X, 1).min() - .5, self.islice_col(X, 1).max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        if X.shape[1] == 3:
            Z = self.yhat(clf, np.c_[xx.ravel(), yy.ravel(), (xx.ravel() * yy.ravel())])
        else:
            Z = self.yhat(clf, np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # plotting
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        if scatter_model is None:
            ax.scatter(self.islice_col(X, 0), self.islice_col(X, 1), **scatter_kwargs)
        else:
            ax.scatter(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                       np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
                       c=scatter_model, **scatter_kwargs)

        # axis
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        if drop_tick_labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        if score is not None:
            ax.text(xx.max() - .2, yy.min() + .2, ('%.2f' % score),
                    dict(position=auc_pos, color='black'), size=auc_size)
        return ax

    def plot_model(self, plot_name='', figure_size=(12, 10), subplot_location=111, fig=None, ax=None, axes_names=None,
                   force_z_lim=None, force_z_max=None, move_by=(0, 0)):

        mod = [self.model[i] for i in [6, 7, 8, 3, 4, 5, 0, 1, 2]]
        return self.plot_interaction(mod, plot_name=plot_name, figure_size=figure_size, fig=fig, ax=ax,
                                     subplot_location=subplot_location, axes_names=axes_names, force_z_lim=force_z_lim,
                                     force_z_max=force_z_max, move_by=move_by)

    @staticmethod
    def plot_interaction(odds, plot_name='', figure_size=(12, 10), subplot_location=111,
                         fig=None, ax=None, axes_names=None, force_z_lim=None, force_z_max=None,
                         move_by=(0, 0), drop_z_ticklabels=False):
        if fig is None:
            fig = plt.figure(figsize=figure_size)
        if ax is None:
            ax = fig.add_subplot(subplot_location, projection='3d')

        xpos = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        ypos = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        zpos = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        dx = np.repeat(0.5, 9)
        dy = np.repeat(0.5, 9)

        dz = odds

        bar_colour_list = ['#2f3744', '#414d5f', '#5e6f88', '#6e7d93', '#8c98aa', '#a3acbb', '#bac1cc', '#d1d5dd',
                           '#e8eaee']
        if force_z_max is None:
            force_z_max = np.array(dz).max()

        bar_colours = [bar_colour_list[8 - int(x)] for x in ((np.array(dz) / force_z_max) * 8)]
        axis_font = {'fontsize': 12}

        this_plot = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=bar_colours, shade=False)
        if force_z_lim is not None:
            ax.set_zlim(force_z_lim[0], force_z_lim[1])
        this_plot.set_edgecolor('white')
        # ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        if plot_name is not None:
            title_font = {'fontsize': 20, 'fontweight': 'medium'}
            ax.set_title(plot_name, fontdict=title_font, y=1.04)
        # ax.set_xlabel('$SNP_1$', fontdict=axis_font, labelpad=10)
        # ax.set_ylabel('$SNP_2$', fontdict=axis_font, labelpad=10)
        if axes_names is None:
            axes_names = ['Locus A', 'Locus B', 'Odds Ratio']
        ax.set_xlabel(axes_names[0], fontdict=axis_font, labelpad=10)
        ax.set_ylabel(axes_names[1], fontdict=axis_font, labelpad=10)
        ax.set_zlabel(axes_names[2], fontdict=axis_font, labelpad=10, **{'rotation': 270})

        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.w_xaxis.line.set_color('#3c3c3c')
        ax.w_yaxis.line.set_color('#3c3c3c')
        ax.w_zaxis.line.set_color('#3c3c3c')
        ax.xaxis._axinfo['tick']['color'] = '#3c3c3c'
        ax.yaxis._axinfo['tick']['color'] = '#3c3c3c'
        ax.zaxis._axinfo['tick']['color'] = '#3c3c3c'
        ax.xaxis.set_ticklabels(['AA', 'Aa', 'aa'])
        ax.yaxis.set_ticklabels(['bb', 'Bb', 'BB'])
        ax.xaxis.set_ticks([1.25, 2.25, 3.25])
        ax.yaxis.set_ticks([1.25, 2.25, 3.25])
        ax.zaxis.set_ticks(np.arange(0, (round(force_z_max * 2) / 2) + 1, 1).tolist())
        if drop_z_ticklabels:
            ax.zaxis.set_ticklabels([])
            ax.zaxis.set_ticks([])

        ax.view_init(azim=-35)

        # Bonus: To get rid of the grid as well:
        ax.grid(False)
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l + move_by[0], b + move_by[1], w, h])

        return ax

    @staticmethod
    def _weighted_random_choice(options, length, probabilities):
        # lifted from here: https://glowingpython.blogspot.com/2012/09/weighted-random-choice.html
        t = np.cumsum(probabilities)
        s = np.sum(probabilities)
        return options[np.searchsorted(t, np.random.rand(length) * s)]

    @staticmethod
    def _append_missing_counts(counts):
        present = [x in counts.index for x in ['00', '10', '01', '11', '02', '20', '12', '21', '22']]
        if sum(present) != 9:
            missing = [not x for x in present]
            extra_counts = pd.Series(np.zeros(sum(missing), dtype=int),
                                     index=list(compress(['00', '10', '01', '11', '02', '20', '12', '21', '22'],
                                                         missing)))
            counts = counts.append(extra_counts)

        return counts

    @staticmethod
    def pprint_vector(vector):
        x = pd.DataFrame(np.reshape(vector, (3, 3)),
                         columns=["bb", "Bb", "BB"],
                         index=["AA", "Aa", "aa"])
        print('Risk alleles: A/B\n')
        print(x)

    @staticmethod
    def scatter_model(name):
        odds_models = dict([('Multiplicative', [1, 1, 1, 1, 2, 2, 1, 2, 2]),
                            ('Threshold', [1, 1, 1, 1, 2, 2, 1, 2, 2]),
                            ('M170 XOR', [1, 2, 1, 2, 1, 2, 1, 2, 1]),
                            ('M78 XOR', [1, 1, 2, 1, 1, 2, 2, 2, 1]),
                            ('M68 Interference', [1, 1, 2, 1, 1, 1, 2, 1, 1])])

        return odds_models.get(name, None)
