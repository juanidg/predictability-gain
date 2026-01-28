#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictability Gain (PG)–based memory estimation for discrete stochastic processes.

This module provides tools to:
- generate synthetic i.i.d. and Markov sequences,
- estimate block entropies using the NSB estimator,
- compute predictability gain components,
- and infer the effective memory order of a sequence using
  a bootstrap hypothesis-testing framework.

The main entry point for users is:

    PG_memory_estimator(S, K, alpha)

which returns the minimal memory order μ that explains the observed
predictability gain structure at significance level alpha.

Author: Juan De Gregorio
Created: Jan 2026

"""
from __future__ import division
import numpy as np
import random as rm
import itertools
import math
import ndd
import numpy.linalg as linalg


def xlog(x: float) -> float:
    """x*log(x) with 0*log(0)=0."""
    return 0.0 if x == 0.0 else x * np.log(x)

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
        - If m>0, returns dict state(tuple) -> p(next_symbol | state).
    seed : int or None

    Returns
    -------
    transProbs : np.ndarray (m=0) OR dict (m>0)
    """
    rng = np.random.default_rng(seed)

    if m == 0:
        return rng.dirichlet(np.ones(L))

    values = range(L)
    states = list(itertools.product(values, repeat=m))  # tuples like (0,1,0)
    return {s: rng.dirichlet(np.ones(L)) for s in states}


def Seq(k, vals, tP, mu, probs=None, T=0, seed=None):
    """
    Generate a discrete sequence from either:
      (i) an i.i.d. process, or
      (ii) an order-mu Markov process.

    Parameters
    ----------
    k : int
        Length of the returned sequence.
    vals : array-like
        Alphabet symbols (e.g. [0, 1]).
    tP : dict or None
        If provided, transition probabilities of an order-mu Markov process:
            tP[state_tuple] = P(next_symbol | state_tuple)
        where state_tuple has length mu.
        If None, the process is assumed i.i.d.
    mu : int or None
        Markov order (required if tP is not None).
    probs : dict
        Probability dictionary:
          - iid case (tP is None):
                probs[symbol] = P(symbol)
          - Markov case (tP is not None):
                probs[state_tuple] = P(initial_state)
    T : int, default=0
        Burn-in time (used only in the Markov case when probs is None).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    seq_out : np.ndarray
        Generated sequence of length k.
    """

    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)

    # ==========================================================
    # Markov case
    # ==========================================================
    if mu is None or mu <= 0:
        raise ValueError("The memory must be greater or equal to 1")

    # --- choose initial state ---
    if probs is None:
        # no initial distribution: start uniformly and apply burn-in
        states = list(tP.keys())
        state = states[rng.integers(len(states))]
        burn_in = int(T)
    else:
        states = list(probs.keys())
        weights = np.asarray(list(probs.values()), dtype=float)
        weights /= weights.sum()
        state = states[rng.choice(len(states), p=weights)]
        burn_in = 0

    seq = list(state)

    # --------------------------
    # Burn-in phase
    # --------------------------
    for _ in range(burn_in):
        sm = tuple(seq[-mu:])
        x = rng.choice(vals, p=tP[sm])
        seq.append(x)

    # --------------------------
    # Recording phase
    # --------------------------
    out = []
    for _ in range(k):
        sm = tuple(seq[-mu:])
        x = rng.choice(vals, p=tP[sm])
        seq.append(x)
        out.append(x)

    return np.asarray(out)


def block_counts(seq, k):
    """
    Count length-k blocks in a sequence using tuple keys.

    Parameters
    ----------
    seq : array-like
        Discrete sequence (symbols).
    k : int
        Block length.

    Returns
    -------
    counts : dict
        counts[block_tuple] = number of occurrences in seq.
    """
    seq = list(seq)
    n = len(seq)
    counts = {}

    # Sliding window of length k
    for i in range(n - k + 1):
        block = tuple(seq[i:i + k])
        counts[block] = counts.get(block, 0) + 1

    return counts


def estimate_H_G(seq, r_list, L):
    """
    Estimate block entropies H_r for r in r_list, and compute predictability gain components:
        G_u = -(H_{u+2} - 2 H_{u+1} + H_u)

    This is the "data-driven" counterpart of the exact computation code.

    Parameters
    ----------
    seq : array-like
        Discrete sequence.
    u_list : iterable of int
        Block lengths to estimate entropies for (e.g., [1,2,3,4,...]).
        To compute G_u for a given u, you need H_u, H_{u+1}, and H_{u+2} to exist.
    L : int
        Alphabet size (used for r = L**block_length in the NSB call).
    estimator : str
        Passed to ndd.entropy. Default "Nsb".

    Returns
    -------
    G_hat : dict
        G_hat[u] = estimated predictability gain component G_u, for u where u,u+1,u+2 are available.
    counts_by_r : dict
        counts_by_r[r] = dict of block counts with tuple keys.
    H_hat : dict
        H_hat[r] = estimated block entropy for block length r (and H_hat[0]=0).
    """
    seq = np.asarray(seq)
    r_list = sorted(set(int(r) for r in r_list))
    counts_by_r = {}
    H_hat = {0: 0.0}

    # --- estimate H_k for each block length k ---
    for r in r_list:
        if r <= 0:
            continue

        counts = block_counts(seq, r)
        counts_by_r[r] = counts

        # counts as a frequency vector for the entropy estimator
        freq = np.asarray(list(counts.values()), dtype=int)

        # NSB (or chosen estimator) over an alphabet of size L**k
        H_hat[r] = ndd.entropy(freq, k=L**r, return_std=False)

    # --- compute predictability gain components from second differences of H ---
    G_hat = {}
    for u in r_list[:-2]:
        G_hat[u] = -(H_hat[u + 2] - 2.0 * H_hat[u + 1] + H_hat[u])

    return G_hat, counts_by_r, H_hat

def est_transP(nblocks, m1, vals, probs):
    """
    Estimate an order-m1 Markov model from empirical block counts.

    This function constructs:
      1) The empirical distribution of length-m1 blocks (used as an
         initial-state distribution), and
      2) The conditional transition probabilities
         P(x_{t+1} | x_{t-m1+1}, ..., x_t).

    Missing or unseen states are handled by assigning them the
    lower-order (i.i.d.) symbol distribution.

    Parameters
    ----------
    nblocks : dict
        Dictionary of block counts:
            nblocks[r][block_tuple] = number of occurrences of a block of length r.
    m1 : int
        Target Markov order.
    vals : array-like
        Alphabet symbols.
    probs : dict
        Empirical single-symbol probabilities (used as fallback for unseen states).

    Returns
    -------
    probs_m : dict
        Empirical distribution of length-m1 blocks:
            probs_m[state_tuple] = P(state_tuple).
    s_transProbs : dict
        Transition probabilities in array form:
            s_transProbs[state_tuple] = list of P(next_symbol | state_tuple),
        ordered according to `vals`.
    """
    # Empirical distribution of length-m1 states
    sum_m = sum(nblocks[m1].values())
    probs_m = {key: value / sum_m for key, value in sorted(nblocks[m1].items())}
    
    # Estimate conditional transition probabilities
    transP = {}
    for st in nblocks[m1]:
        transP[st] = {}
        for v in vals:
            transP[st][v] = nblocks[m1+1].get(st + (v,), 0)
        suma = sum(transP[st].values())
        if suma == 0:
            del transP[st]
            continue    
        for v in transP[st]:
            transP[st][v] /= suma

    cats = sorted(vals)
    
    # States observed in the data
    s_transProbs = {st: list([transP[st][a] for a in vals]) for st in transP}
    
    # Add missing states and assign them iid probabilities
    for st in itertools.product(cats, repeat=m1):
        if st not in s_transProbs:
            s_transProbs[st] = list(probs.values())
            
    return probs_m, s_transProbs

def fisher_pval(pvals,mu,rmax):
    """
    Combine independent p-values using Fisher's method.
    
    """
    z = np.prod(pvals)
    if z==0:
        return 0
    pv = z*sum((-np.log(z))** j / math.factorial(j) for j in range(rmax-mu-1))
    return pv


def compute_G(transProbs, m, L, tol=1e-10, warn=True):
    """
    Compute exact block entropies H_r and predictability gains G_u
    from a fully specified process.

    """

    states = list(transProbs.keys())
    # ensure numpy arrays
    transProbs = {s: np.asarray(transProbs[s], dtype=float) for s in states}
    
    nmax = m + 1
    values = range(L)


    S = len(states)
    idx = {s: i for i, s in enumerate(states)}

    # Build transition matrix T over length-m states
    # column = current state, row = next state:
    # T[next_state, current_state] = P(a | current_state)
    T = np.zeros((S, S))

    for s in states:
        i = idx[s]
        suf = s[1:]            # tuple of length m-1
        ps = transProbs[s]     # length-L prob vector

        for a in range(L):
            t = suf + (a,)     # append symbol a to make next tuple-state
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
    
    return G

        
def PG_memory_estimator(S,K,alpha):
    
    """
    Estimate the effective memory order of a discrete sequence using
    predictability gain (PG) and bootstrap hypothesis testing.

    The procedure tests increasing candidate memory orders μ = 0, 1, 2, ...
    and returns the smallest μ for which an order-μ Markov model is sufficient
    to reproduce the observed predictability gain at longer conditioning lengths.

    Operationally, for each μ:
      1. Generate K surrogate sequences from an order-μ model inferred from data.
      2. Compute their predictability gain components.
      3. Compare them to the observed PG values.
      4. Combine p-values across lags using Fisher's method.
      5. Stop at the first μ for which the null hypothesis is not rejected.

    Parameters
    ----------
    S : array-like
        Observed discrete sequence.
    K : int
        Number of bootstrap surrogate sequences.
    alpha : float
        Significance level of the test.

    Returns
    -------
    m_pg : int or str
        Estimated memory order, or a warning string if estimation fails.
    """
  
    N = len(S)
    values = sorted(set(S))
    L = len(values)
    if L == 1:
         return 'Warning: the sequence only has one outcome'
     
    # Maximum block length allowed by data size           
    rmax = int(np.floor(np.log(N)/np.log(L)))
    if rmax < 2:
        return 'Warning: Insufficient data'
    
    # Conditioning lengths and candidate memory orders
    r_list = list(range(rmax+1))
    mu_list= r_list[:-2]
    
    # Estimate PG from the observed sequence
    G_est, counts_r, H_est = estimate_H_G(S, r_list, L)
    probs = {key: value / N for key, value in sorted(counts_r[1].items())}
    
    # Test increasing candidate memory orders μ
    for mu in mu_list:
        cases = {j: 0 for j in range(mu,rmax-1)}
        if mu > 0:
            probs_m, tP_m = est_transP(counts_r, mu, values,probs)
        # ---------------------------------------------
        # Bootstrap loop
        # ---------------------------------------------
        for k in range(K):
            
            # Generate surrogate sequences
            if mu==0:
                # i.i.d. surrogate
                s = rm.choices(values,weights=probs.values(),k=N)
            else:
                # Markov surrogate of order mu
                s = Seq(k=N, vals=values, tP=tP_m, mu=mu, probs=probs_m)
           
            # Estimate PG for surrogate sequence
            Gm = estimate_H_G(s, r_list[mu:], L)[0]
            
            # Compare surrogate PG to observed PG
            for m2 in mu_list[mu:]:
                if abs(Gm[m2]) >= abs(G_est[m2]):
                    cases[m2] += 1
                    
        # Combine p-values using Fisher's method
        pvals = np.fromiter(cases.values(), dtype=int)/K
        pval_f = fisher_pval(pvals, mu, rmax)
        
        # Stop at the smallest mu that is not rejected
        if pval_f >= alpha:
            m_pg = mu
            return m_pg
    return 'No memory detected within the proposed range'
    
# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
# NOTE:
# The memory estimator can be applied directly to any discrete sequence `S`
# provided by the user (e.g. empirical data).
#
# Example (real data):
# m_pg = PG_memory_estimator(S_observed, K=2000, alpha=0.05)
#
# The synthetic process and sequence generation below are included
# *only for demonstration and tests purposes*, to illustrate how the estimator
# behaves on data with a known memory order.
# -------------------------------------------------------------------
if __name__ == "__main__":   
    
    # -------------------------------------------------
    # Parameters for the synthetic example
    # -------------------------------------------------
    N = 100        # sequence length
    m = 1          # true memory order of the synthetic process
    L = 2          # alphabet size
    vals = list(range(L))
    K = 2000       # number of bootstrap surrogates
    alpha = 0.05   # significance level
    
    # -------------------------------------------------
    # Generate a random synthetic process
    # -------------------------------------------------
    transProbs = random_transprobs(L=L,m=m,seed=None)
    
    # -------------------------------------------------
    # Generate a sequence from the process
    # -------------------------------------------------
    if m==0:
        S = rm.choices(vals,weights=transProbs,k=N)
    else:
        # Optional sanity check:
        # verify that the predictability gain at m-1 is not too small
        G = compute_G(transProbs, m, L, tol=1e-10, warn=True)
        if G[-1]<=0.04:
            print('$G_{m-1} < 0.04$')
        S = Seq(k=N, vals=vals, tP=transProbs, mu=m, probs=None, T=100, seed=None)
    
    # -------------------------------------------------
    # Estimate memory from the observed sequence
    # -------------------------------------------------
    m_pg = PG_memory_estimator(S, K, alpha)
    
    print("Estimated memory:", m_pg)
    



















