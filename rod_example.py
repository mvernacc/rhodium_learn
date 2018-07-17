"""Rod example of Monte Carlo methods for engineering uncertainty"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import rhodium as rdm
import seaborn as sns

from quantile_plot import quantile_plot


# Assume the rod strength is normally distributed
strength_mean = 100.
strength_sd = 5.

# Assume the max. load is normally distributed
load_mean = 85.
load_sd = 10.

###################
# Analytic solution
###################

# Compute the parameters of the margin (strength - load) distribution.
m_mean = strength_mean - load_mean
m_sd = (strength_sd**2 + load_sd**2)**0.5

m_rv = stats.norm(loc=m_mean, scale=m_sd)

# Compute the probability of failure from the cdf
p_fail = m_rv.cdf(0)
print('Analytic failure probability: {:.4f}'.format(p_fail))


###############
# Rhodium model
###############

# Define the model
def rod_model(strength=strength_mean, load=load_mean):
    margin = strength - load
    return margin

model = rdm.Model(rod_model)

model.parameters = [
    rdm.Parameter('strength'),
    rdm.Parameter('load'),
]

model.responses = [
    rdm.Response('margin', rdm.Response.MAXIMIZE)
]

model.uncertainties = [
    rdm.NormalUncertainty('strength', mean=strength_mean, stdev=strength_sd),
    rdm.NormalUncertainty('load', mean=load_mean, stdev=load_sd),
]

# Sample scenarios from the input distribution
scenarios = rdm.sample_lhs(model, nsamples=1000)
# Evaluate the model at each  scenario
results = rdm.evaluate(model, scenarios)

m_samples = np.array(results['margin'])

# Compute the failure probability
p_fail_mc = sum(m_samples < 0) / len(m_samples)
print('Monte-Carlo failure probability: {:.4f}'.format(p_fail_mc))


####################################################
# Compare analytic and MC solutions
####################################################
sns.distplot(results['margin'], label='Monte Carlo, $p_{{fail}}={:.4f}$'.format(p_fail_mc))

x = np.linspace(m_mean - 4 * m_sd, m_mean + 4 * m_sd, 100)
plt.plot(x, m_rv.pdf(x), linestyle='--', color='black',
    label='Analytic, $p_{{fail}}={:.4f}$'.format(p_fail))
plt.fill_between(x[x <= 0], 0, m_rv.pdf(x[x <= 0]), facecolor='red', alpha=0.5)
plt.axvline(x=0, color='red')
plt.title('Margin')
plt.ylabel('Prob. density')
plt.legend()
plt.ylim([0, plt.ylim()[1]])


######################################
# More plots from the Monte Carlo data
######################################
# One-factor-at-a-time sensitivity plot, aka tornado plot
rdm.oat(model, 'margin')

# "Upside-downside plot" from de Neufville 
plt.figure()
ax1 = plt.subplot(1, 2, 1)
quantile_plot(results['strength'], results['margin'], scatter=True, ax=ax1)
plt.xlabel('Strength')
plt.ylabel('Margin')
plt.axhline(y=0, color='red')
plt.legend()

ax2 = plt.subplot(1, 2, 2, sharey=ax1)
quantile_plot(results['load'], results['margin'], scatter=True, ax=ax2)
plt.xlabel('Load')
plt.ylabel('Margin')
plt.axhline(y=0, color='red')
plt.legend()

plt.show()
