import numpy as np
import numpy.linalg as la 
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st


def enforce_corner_constraint(theta, n_teams):
    theta[0] = theta[n_teams] = 0
    return theta

def propose_samples(current, sigma):
    I = np.eye(current.shape[0])
    proposal = np.random.multivariate_normal(current, cov=I*sigma**2)
    return proposal

def compute_hyper_prior_probabilities(eta):
    tau1 = 0.0001
    alpha = beta = 0.1

    mu_attack_logprob = st.norm(0, 1/np.sqrt(tau1)).logpdf(eta[0])
    mu_defense_logprob = st.norm(0, 1/np.sqrt(tau1)).logpdf(eta[1])
    tau_attack_logprob = st.gamma(alpha, scale=1/beta).logpdf(eta[2])
    tau_defense_logprob = st.gamma(alpha, scale=1/beta).logpdf(eta[3])

    eta_logprob = mu_attack_logprob + mu_defense_logprob + tau_attack_logprob + tau_defense_logprob
    return eta_logprob

def compute_prior_probabilities(eta, theta, n_teams):
    tau0 = 0.0001

    mu_attack = eta[0]
    mu_defense = eta[1]
    tau_attack = eta[2]
    tau_defense = eta[3]
    
    home = theta[-1]
    attack = theta[:n_teams]
    defense = theta[n_teams:-1]

    home_logprob = st.norm(0, 1/np.sqrt(tau0)).logpdf(home)
    attack_logprob = st.norm(mu_attack, 1/np.sqrt(tau_attack)).logpdf(attack)
    defense_logprob = st.norm(mu_defense, 1/np.sqrt(tau_defense)).logpdf(defense)

    theta_prob = home_logprob + np.sum(attack_logprob + defense_logprob)
    return theta_prob

def compute_likelihood(theta, n_teams, dataf):
    
    goals_home = dataf['goals_home'].values.astype(int)
    goals_away = dataf['goals_away'].values.astype(int)
    home_team = dataf['home_team'].values.astype(int)
    away_team = dataf['away_team'].values.astype(int)

    home = theta[-1]
    attack = theta[:n_teams]
    defense = theta[n_teams:-1]

    theta_home = np.exp(home + attack[home_team] - defense[away_team])
    theta_away = np.exp(attack[away_team] - defense[home_team])
    
    loglikelihood_home = st.poisson(theta_home).logpmf(goals_home)
    loglikelihood_away = st.poisson(theta_away).logpmf(goals_away)

    loglikelihood = np.sum(loglikelihood_home + loglikelihood_away)
    return loglikelihood

def compute_probabilities(eta, theta, n_teams, dataf):
    eta_logprob = compute_hyper_prior_probabilities(eta)
    theta_logprob = compute_prior_probabilities(eta, theta, n_teams)
    loglikelihood = compute_likelihood(theta, n_teams, dataf)
    logprob = eta_logprob + theta_logprob + loglikelihood
    return logprob

def metropolis_hastings(n_samples, n_teams, sigma, dataf, thinning, burn_in, sample_burn_in=False):

    starting_point = 0.1
    eta_current = np.full(4, starting_point)
    theta_current = enforce_corner_constraint(np.full(1+n_teams*2, starting_point), n_teams)
    samples = []
    rejection = []
    t = 0
    
    if sample_burn_in:
        n_samples = n_samples+(burn_in/thinning)

    while n_samples > len(samples):

        eta_proposal = propose_samples(eta_current, sigma)
        theta_proposal = enforce_corner_constraint(propose_samples(theta_current, sigma), n_teams)

        current_logprob = compute_probabilities(eta_current, theta_current, n_teams, dataf)
        proposal_logprob = compute_probabilities(eta_proposal, theta_proposal, n_teams, dataf)
        acceptance_logprob = proposal_logprob - current_logprob

        u = np.random.uniform()
        if np.log(u) < acceptance_logprob:
            eta_current = eta_proposal.copy()
            theta_current = theta_proposal.copy()
            rejection.append(0)
        else:
            rejection.append(1)

        if t % thinning == 0:
            if sample_burn_in:
                samples.append(theta_current.copy())     
            elif t > burn_in:
                samples.append(theta_current.copy())     

        t += 1   
        print(f'\r{len(samples)} / {n_samples} samples', end='')

    return np.array(samples), np.array(rejection)


dataf = pd.read_csv('premier_league_2013_2014.dat', sep=',', header=None)
dataf.columns = ['goals_home', 'goals_away', 'home_team', 'away_team']


n_samples = 5000
n_teams = 20
sigmas = [0.005, 0.05, 0.5]
thinning = [1, 5, 20, 50]
burn_in = 5000
home_batches = []
rejection_batches = []

for i in range(len(sigmas)):
    for j in range(len(thinning)):

        print(f'\nRunning sigma={sigmas[i]} and thinning={thinning[j]}')
        samples, rejection = metropolis_hastings(n_samples, n_teams, sigmas[i], dataf, thinning[j], burn_in, sample_burn_in=True)
        homes = samples[:,-1]
        home_batches.append(homes)
        rejection_batches.append(rejection)


plot_its = burn_in + n_samples
t = 0
for i in range(len(sigmas)):
    for j in range(len(thinning)):

        rejection = rejection_batches[t]
        rejection = rejection[burn_in:]
        print(f'Rejection rate: sigma={sigmas[i]}, thinning={thinning[j]}:', np.round(np.mean(rejection), 4))

        home = home_batches[t]
        plot_samples = plot_its//thinning[j]
        its = np.array(np.linspace(0,plot_its,plot_samples))
        n_homes = home.shape[0]

        fig = plt.figure()
        ax = plt.axes()
        ax.plot(its, home[:plot_samples], '-')
        # ax.plot(np.array(np.linspace(0,n_homes*thinning[j],n_homes)), home, '-')
        ax.axvline(x=burn_in, color='red', linestyle='--', label='Burn-in')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('$home$')
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        plt.show()

        t += 1


samples, _ = metropolis_hastings(n_samples, n_teams, sigmas[1], dataf, thinning[1], burn_in, sample_burn_in=False)
homes = samples[:,-1]
proposal_x = np.linspace(np.min(homes), np.max(homes), 1000)
proposal_y = st.norm(np.mean(homes), sigmas[1]).pdf(proposal_x)

fig = plt.figure()
plt.hist(homes, bins=50, density=True, label='Posterior distribution')
plt.plot(proposal_x, proposal_y, color='red', label='Proposal distribution')
plt.xlabel('$home$')
plt.ylabel('PDF')
plt.grid(True, alpha=0.25)
plt.legend()
fig.tight_layout()
plt.show()


expected_attack = np.mean(samples[:,:n_teams], 0)
expected_defense = np.mean(samples[:,n_teams:-1], 0)

fig = plt.figure()
ax = plt.axes()
ax.plot(expected_defense, expected_attack, 'o')
for i in range(expected_defense.shape[0]):
    ax.text(expected_defense[i], expected_attack[i], str(i))
ax.set_xlabel(r'$E_{p(\theta,\eta\mid y)}[def_i]$')
ax.set_ylabel(r'$E_{p(\theta,\eta\mid y)}[att_i]$')
ax.grid(True, alpha=0.25)
fig.tight_layout()
plt.show()