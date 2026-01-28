#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact block entropies H_r and predictability gain components G_u
for a discrete stochastic process.

Input format:
- i.i.d. case (m = 0): transProbs is a length-L vector p(x).
- Markov case (m > 0): transProbs is a dict mapping each length-m state (string)
  to a length-L vector.

We compute:
- Block entropies H_r for r = 0..nmax, where nmax = m + 3
- Predictability gain components:
      G_u = -(H_{u+2} - 2 H_{u+1} + H_u),   u = 0..nmax-2
- The entropy-rate slope h and a linear asymptote line used in plots.
"""
import itertools
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Helper: x log x with the convention 0 log 0 = 0
# -------------------------------------------------------------------
def xlog(x: float) -> float:
    """x*log(x) with 0*log(0)=0."""
    return 0.0 if x == 0.0 else x * np.log(x)

# -------------------------------------------------------------------
# Random generator for a process specification (optional helper)
# -------------------------------------------------------------------
def random_transprobs(L: int, m: int, seed=None):
    """
    Create transition probabilities with random entries.

    Parameters
    ----------
    L : int
        Alphabet size.
    m : int
        Markov order.
        - If m=0, returns a length-L iid distribution p(x).
        - If m>0, returns dict state -> p(next_symbol | state).
    seed : int or None

    Returns
    -------
    transProbs : np.ndarray (m=0) OR dict (m>0)
    """
    rng = np.random.default_rng(seed)

    if m == 0:
        return rng.dirichlet(np.ones(L))

    values = range(L)
    states = ["".join(str(e) for e in item) for item in itertools.product(values, repeat=m)]
    return {s: rng.dirichlet(np.ones(L)) for s in states}

def compute_H_G(transProbs, tol=1e-10, warn=True):
    """
    Compute exact block entropies H_r and predictability gains G_u
    from a fully specified process.

    Parameters
    ----------
    transProbs :
        - m = 0 (iid): 1D array-like of length L, p(x)
        - m > 0: dict mapping length-m state strings to
          length-L arrays p(x | state)

    Returns
    -------
    H : list
        Block entropies H_r for r = 0..nmax
    G : list
        Predictability gain components G_u for u = 0..nmax-2
    line : ndarray
        Linear asymptote used for plotting
    slope : float
        Entropy-rate slope
    """

    # -----------------------
    # Infer m and L
    # -----------------------
    if isinstance(transProbs, dict):
        states = list(transProbs.keys())
        m = len(states[0])
        L = len(transProbs[states[0]])
    else:
        p = np.asarray(transProbs, dtype=float)
        m = 0
        L = len(p)

    nmax = m + 3
    values = range(L)

    # =====================================================
    # Case m = 0 (iid)
    # =====================================================
    if m == 0:
        H1 = -sum(xlog(pi) for pi in p)
        H = [0.0] + [r * H1 for r in range(1, nmax + 1)]
        G = [0.0 for _ in range(nmax - 1)]
        slope = H1
        n_list = np.arange(0, nmax + 1)
        line = slope * n_list
        return H, G, line, slope

    # =====================================================
    # Case m > 0 (Markov)
    # =====================================================
    # Ensure values are numpy arrays (user may provide Python lists)
    transProbs = {s: np.asarray(transProbs[s], dtype=float) for s in states}
    
    S = len(states)
    idx = {s: i for i, s in enumerate(states)}

    # Build transition matrix T over length-m states
    # From current_state = (x_1,...,x_m), after drawing next symbol a,
    # the next_state = (x_2,...,x_m,a).
    #
    # We store T as column-stochastic: column = current state, row = next state:
    #     T[next_state, current_state] = P(a | current_state)
    
    T = np.zeros((S, S))
    for s in states:
        i = idx[s]
        suf = s[1:]
        ps = transProbs[s]          # vector p(a | s)
        for a in range(L):
            t = suf + str(a)        # next state after appending a
            j = idx[t]
            T[j, i] = ps[a]

    # Stationary distribution
    eigvals, eigvecs = linalg.eig(T)

    mult = int(np.sum(np.abs(eigvals - 1.0) < tol))
    if warn and mult > 1:
        print(
            "Warning: eigenvalue 1 has multiplicity > 1. "
            "Stationary distribution is not unique."
        )

    k = int(np.argmin(np.abs(eigvals - 1.0)))
    P = np.real(eigvecs[:, k])
    P /= P.sum()

    # Entropy-rate slope
    c = [sum(xlog(px) for px in transProbs[s]) for s in states]
    slope = -np.dot(c, P)

    # -----------------------------------------------------
    # Exact block probabilities probs[r] for r = 1..nmax
    #
    # For r < m: marginalize the stationary distribution over length-m states.
    # For r = m: probs[m] is exactly P over length-m blocks.
    # For r > m: extend blocks recursively:
    # P(x_1,..,x_r) = P(x_1,..,x_{r-1}) * P(x_r | last m symbols of prefix)
    # -----------------------------------------------------
    probs = {}

    # r < m
    for r in range(1, m):
        pi = []
        j = 0
        step = L ** (m - r)
        for _ in range(L ** r):
            pi.append(np.sum(P[j:j + step]))
            j += step
        probs[r] = pi

    # r = m
    probs[m] = list(P)

    # flatten conditional probs
    TP2 = []
    for s in states:
        TP2.extend(transProbs[s])

    # r > m
    for r in range(m + 1, nmax + 1):
        TPP = TP2 * (L ** (r - m))
        PP = []
        for pr in probs[r - 1]:
            for _ in values:
                PP.append(pr)
        probs[r] = [a * b for a, b in zip(TPP, PP)]

    # -----------------------------------------------------
    # Block entropies
    # -----------------------------------------------------
    H = [0.0]
    for r in range(1, nmax + 1):
        H.append(-sum(xlog(p) for p in probs[r]))

    # Predictability gains
    G = []
    for u in range(nmax - 1):
        G.append(-(H[u + 2] - 2 * H[u + 1] + H[u]))

    n_list = np.arange(0, nmax + 1)
    line = slope * (n_list - m) + H[m]

    return H, G, line, slope

def plot_H_G(H, G, line):
    """Plot H_r and G_u."""
    nmax = len(H) - 1
    n_list = list(range(0, nmax + 1))
  
    SMALL_SIZE = 19
    MEDIUM_SIZE = 26
    BIGGER_SIZE = 29
    
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=19)    # legend fontsize
    #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure(figsize=(8, 6))
    plt.plot(n_list, np.array(H), "k.--", markersize=17, label=r"$H_r$")
    plt.plot(n_list, line, "k-", label=r"$\mathcal{H}(r)$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$H_r$", labelpad=10)
    plt.xlim(-0.3, n_list[-1] + 0.3)
    plt.xticks(np.arange(0, n_list[-1] + 1, 1.0))
    plt.margins(x=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(n_list[:-2], np.array(G), "r.--", markersize=17)
    plt.axhline(0)
    plt.xlabel(r"$u$")
    plt.ylabel(r"$\mathcal{G}_u$", labelpad=10)
    plt.xlim(-0.3, n_list[-2] - 0.3)
    plt.xticks(np.arange(0, n_list[-2], 1.0))
    plt.margins(x=0)
    plt.tight_layout()
    plt.show()
    
# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    transProbs = {
        '000': [0.07311298932500401, 0.926887010674996],
        '001': [0.9622449051926358, 0.03775509480736413],
        '010': [0.5125802722131408, 0.4874197277868591],
        '011': [0.8377024292375447, 0.16229757076245538],
        '100': [0.5908515367157229, 0.40914846328427706],
        '101': [0.16170469315464595, 0.838295306845354],
        '110': [0.2615494592422908, 0.7384505407577092],
        '111': [0.3972880723117724, 0.6027119276882276]
    }
    
    # Uncomment next line for random transProbs
    #transProbs = random_transprobs(L=2,m=2,seed=None)


    H, G, line, slope = compute_H_G(transProbs)
    plot_H_G(H, G, line)
