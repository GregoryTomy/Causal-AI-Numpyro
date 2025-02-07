from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

import jax.numpy as jnp
import jax.random as random

import numpyro as nr
from numpyro.infer import Predictive
import numpyro.distributions as dist

# Listing 2.1: Implementing a discrete distribution table in pgmpy

discrete_dist = DiscreteFactor(
    variables=["X"],
    cardinality=[3],
    values=[0.45, 0.30, 0.25],
    state_names={"X": ["G", "2", "3"]},
)

print(discrete_dist)

# Listing 2.2 Modeling a joint distribution in pgmpy

joint_dist = DiscreteFactor(
    variables=["X", "Y"],
    cardinality=[3, 2],
    values=[0.25, 0.2, 0.2, 0.1, 0.15, 0.1],
    state_names={"X": ["1", "2", "3"], "Y": ["0", "1", "2"]},
)
print(joint_dist)

print(joint_dist.marginalize(variables=["Y"], inplace=False))
print(joint_dist.marginalize(variables=["X"], inplace=False))

# conditional distribution
# P(Y|X) = P(X, Y) / P(X)
print(joint_dist / discrete_dist)

# Defining a CPD(COnditional probaility distribution)

cpd_Y_given_X = TabularCPD(
    variable="Y",  # The child variable (dependent variable)
    variable_card=2,  # Y has two possible states
    values=[
        [0.25 / 0.45, 0.2 / 0.3, 0.15 / 0.25],  # P(Y=0 | X)
        [0.2 / 0.45, 0.1 / 0.3, 0.1 / 0.25],  # P(Y=1 | X)
    ],
    evidence=["X"],  # Parent variable
    evidence_card=[3],  # X has 3 possible states
)

print(cpdGg_Y_given_X)

# Listing 2.3 Canonical parameters in Numpyro
#
print(dist.Categorical(probs=jnp.array([0.45, 0.30, 0.25])))
print(dist.Normal(0.0, 1.0))
print(dist.Bernoulli(probs=0.4))
print(dist.Gamma(concentration=1.0, rate=2.0))

bernoulli = dist.Bernoulli(0.4)

bernoulli_log = bernoulli.log_prob(1.0)
jnp.exp(bernoulli_log)

# Listing 2.4 Simulating from DiscreteFactor in pgmpy and Numyro

# simulating from pgmpy

discrete_dist.sample(n=2)

joint_dist.sample(n=2)

key = random.PRNGKey(0)
dist.Categorical(probs=jnp.array([0.45, 0.30, 0.25])).sample(key, sample_shape=(10,))

categorical_dist = dist.Categorical(probs=jnp.array([0.45, 0.30, 0.25]))
nr.sample("cat", categorical_dist, rng_key=key, sample_shape=(10,))


# Listing 2.5 Creating a random process in pgmpy and numpyro.
# z ~ P(Z)
# x ~ P(X | Z = z)
# y ~ P(Y | X = x)

prob_z = TabularCPD(variable="Z", variable_card=2, values=[[0.65], [0.35]])

prob_x_given_z = TabularCPD(
    variable="X",
    variable_card=2,
    values=[
        [0.8, 0.6],
        [0.2, 0.4],
    ],
    evidence=["Z"],
    evidence_card=[2],
)

prob_y_given_z = TabularCPD(
    variable="Y",
    variable_card=3,
    values=[
        [0.1, 0.8],
        [0.2, 0.1],
        [0.7, 0.1],
    ],
    evidence=["X"],
    evidence_card=[2],
)

print(prob_y_given_z)


model = BayesianNetwork([("Z", "X"), ("X", "Y")])
print(model)

model.add_cpds(prob_z, prob_x_given_z, prob_y_given_z)

generator = BayesianModelSampling(model)
generator.forward_sample(2)

# Listing 2.6 Working with combinations of canonical distributions in numpyro

# Use NumPyro's handler to seed randomness (since JAX doesn't have global random state)
with nr.handlers.seed(rng_seed=0):
    N = 10
    z = nr.sample("z", dist.Gamma(0.75, 1.0).expand([N]))
    x = nr.sample("x", dist.Poisson(z))
    p = x / (5 + x)
    y = nr.sample("y", dist.Bernoulli(p))
    print(z, x, y)


# Listing 2.7 Random processes with nuanced control flow in NumPyro
def random_process_2():
    z = nr.sample("z", dist.Gamma(7.5, 1.0))
    x = nr.sample("x", dist.Poisson(z))

    # Note: the sum of y indpendent Bernoulli(0.5) trials
    # follows a bionmial distribution
    y = nr.sample("y", dist.Binomial(x, 0.5))

    return y


# generating samples pgmpy
pgmpy_samples = generator.forward_sample(100)
print(pgmpy_samples.mean())

# generating smaples using numpyro

# generate samples
predictive = Predictive(random_process_2, num_samples=100)
samples = predictive(key)
generated_samples = samples["y"]
generated_samples.mean()

jnp.square(generated_samples).mean()


# Listing 2.9 Monte Carlo estimation in Numpyro
def random_process_3():
    z = nr.sample("z", dist.Gamma(7.5, 1.0))
    x = nr.sample("x", dist.Poisson(z))

    # Note: the sum of y indpendent Bernoulli(0.5) trials
    # follows a bionmial distribution
    y = nr.sample("y", dist.Binomial(x, 0.5))

    return y, z


predictive_3 = Predictive(random_process_3, num_samples=1000)
samples = predictive_3(random.PRNGKey(7))
samples_z = samples["z"]
z_mean = samples_z.mean()


# Estimating E(Z|Y=3)
z_given_y = jnp.stack([z for z, y in zip(samples["z"], samples["y"]) if y == 3])
print(z_given_y.mean())

# Listing 2.10 Generating IID samples in Numpyro


def random_process_4(num_samples: int):
    z = nr.sample("z", dist.Gamma(7.5, 1.0))
    x = nr.sample("x", dist.Poisson(z))

    # Note: the sum of y indpendent Bernoulli(0.5) trials
    # follows a bionmial distribution
    with nr.plate("plate_y", num_samples):
        y = nr.sample("y", dist.Binomial(x, 0.5))

    return y


# nr.handlers.seed() wraps the function, ensuring all samples
# inside use the same fixed random seed
with nr.handlers.seed(rng_seed=10):
    samples = random_process_4(10)
    print(samples)


# Listing 2.11 An example of a DGP in code form
def true_dgp(jenny_inclination, brian_inclination, window_strength):
    jenny_throws_rock = jenny_inclination > 0.5
    brian_throws_rock = brian_inclination > 0.5
    if jenny_throws_rock and brian_throws_rock:
        strength_of_impact = 0.81
    elif jenny_throws_rock or brian_throws_rock:
        strength_of_impact = 0.6
    else:
        strength_of_impact = 0.0
    window_breaks = window_strength < strength_of_impact
    return jenny_throws_rock, brian_throws_rock, window_breaks


# the parameters are latent variables between 0 and 1
# the return values are the observed variables in the data

initials = [
    (0.6, 0.31, 0.83),
    (0.48, 0.53, 0.33),
    (0.66, 0.63, 0.75),
    (0.65, 0.66, 0.8),
    (0.48, 0.16, 0.27),
]

data_points = []
for jenny_inclination, brian_inclination, window_strength in initials:
    data_points.append(true_dgp(jenny_inclination, brian_inclination, window_strength))

data_points
