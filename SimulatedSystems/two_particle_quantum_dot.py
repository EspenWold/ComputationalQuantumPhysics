from math import sqrt, exp
import numpy as np

def q_dot_wf_squared(r, variational_params):
    (alpha, beta) = variational_params
    r1 = r[0] ** 2 + r[1] ** 2
    r2 = r[2] ** 2 + r[3] ** 2
    r12 = sqrt((r[0] - r[2]) ** 2 + (r[1] - r[3]) ** 2)
    deno = r12 / (1 + beta * r12)
    return exp(-0.5 * alpha * (r1 + r2) + deno)**2


def q_dot_local_energy(r, variational_params):
    (alpha, beta) = variational_params
    r1 = (r[0] ** 2 + r[1] ** 2)
    r2 = (r[2] ** 2 + r[3] ** 2)
    r12 = sqrt((r[0] - r[2]) ** 2 + (r[1] - r[3]) ** 2)
    deno = 1.0 / (1 + beta * r12)
    deno2 = deno * deno
    return 0.5 * (1 - alpha * alpha) * (r1 + r2) + 2.0 * alpha + 1.0 / r12 + deno2 * (
            alpha * r12 - deno2 + 2 * beta * deno - 1.0 / r12)


def q_dot_q_force(r, variational_params):
    (alpha, beta) = variational_params
    qforce = np.zeros(r.shape, np.double)
    r12 = sqrt((r[0]-r[2])**2 + (r[1]-r[3])**2)
    deno = 1.0/(1+beta*r12)
    for i in range(len(r)):
        qforce[i] = -2*r[i]*alpha*(r[i]-r[(i+2) % 4])*deno*deno/r12
    return qforce
