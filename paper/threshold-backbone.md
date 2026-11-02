# The Threshold Parameter in Robust Losses: What the Annealing Rule Is, What It Costs, and What It Buys

> Working backbone for a TSP submission, v6. Sections 1–8 are the independent work; introduction,
> prior work and references are deliberately absent. Status: theory settled; one controlled solver
> experiment done; benchmark comparison deferred.
>
> **Changes from v5.** Three arithmetic errors found by external review, and three internal
> inconsistencies.
> (i) Prop. 4.3: the last line of the *proof* dropped a factor $\nu$; the statement was correct.
> (ii) Prop. 4.2: the intermediate display read $\Phi(-t)-\Phi(-t)^2$ where it should read
> $2[\Phi(-t)-\Phi(-t)^2]$; the value $1/\sqrt\pi$ was correct.
> (iii) Prop. 5.2 table: the floor had been set to the Gumbel centring constant $2.610$ instead of
> $b_{200}=2.746$; the whole table is regenerated. This also makes the $\varepsilon=0$ row consistent,
> since a floor at $-b_K$ can only bind below the clean mean.
> (iv) Contribution C3 no longer calls $1/K$ a breakdown point, matching Prop. 5.1.
> (v) Q3 no longer promises a multi-threshold law.
> (vi) C3 drops the cross-dimensional comparison.
> (vii) New §5.1: the threshold path is a robustness *path*, in the sense that it monotonically
> tightens the rejection point; this is where the paper meets classical robust estimation.
> (viii) §7.1 reconstruction figures restated from the current runs.

---

## 1. The substrate and the rule

### 1.1 Setting

A robust loss family $\{\varphi_\theta\}_{\theta>0}$ is applied to residuals $r=A(x)-y$ of a
low-rank or structured recovery problem, and the parameter $\theta$ is the point at which the loss
stops increasing. Recovery alternates between a state update at fixed $\theta$ and a threshold
update from the current residuals:

$$
\theta_k=\min\{\hat\theta_k,\ \theta_{k-1}\},\qquad
\hat\theta_k=T(r_k),\qquad r_k=A(x_k)-y . \tag{1.1}
$$

A family of published methods uses (1.1) in this exact form: hybrid ordinary–Welsch (HOW) matrix
recovery, hybrid quadratic–Frobenius (HQF) matrix recovery, $\ell_{p,\varepsilon}$ tensor
completion, capped-Frobenius tensor completion, and robust elliptic localisation. The statistic $T$
changes — median absolute deviation from the median, normalised (MADN), with $\zeta=3$; normalised
interquartile range; an $\ell_\infty$ convexity bound — the rule does not. In all five the threshold
is a single scalar; none of them carries one threshold per tensor mode or per residual channel.

In every case the rule is justified by one computation, and the computation is the same in all
five:

$$
\frac{\partial}{\partial\theta}\,\mathcal L_\theta(x)\ \ge\ 0
\quad\Longrightarrow\quad
\theta_k\le\theta_{k-1}\ \Rightarrow\ \mathcal L_{\theta_k}(x_k)\le\mathcal L_{\theta_{k-1}}(x_k). \tag{1.2}
$$

Call (1.2) the **descent certificate**. It certifies *any* non-increasing schedule, it ranks none,
and it says nothing about what the schedule converges to.

**Two requirements are being run together in the literature, and we keep them apart.** (1.2)
certifies monotonicity, and monotonicity alone. Whether $\theta_k$ is a sensible value for the
residuals in front of it is a separate **licence constraint** $y_k\le\hat\theta_k$, not implied
by (1.2): a schedule may be non-increasing and still sit far above every threshold the residuals
support, and (1.2) will still certify it. The quantity measured against the licence constraint
is **over-trust**; the quantity measured against the target is **bias**. The published rule
minimises bias among the schedules that never breach the licence (Corollary 2.6); the over-trust
is what it would have to spend to do better.

Over-trust is a technical quantity of the threshold layer and carries no evaluative charge of its
own. Spending it can hurt recovery or help it, depending on where the retained observations lie
(§7.5). Nothing below should be read as "less over-trust is better recovery"; the frontier prices a
trade, it does not recommend an outcome.

### 1.2 The question

Fix a block of $K$ iteration steps and write $\hat\theta_{1:K}$ for the unconstrained estimates.
The certificate (1.2) defines an admissible set; the rule (1.1) is one point of it. The questions
this paper answers are:

* **(Q1) Characterisation.** What is (1.1) as an object in the admissible set?
* **(Q2) Frontier.** What is the set of (bias, over-trust) pairs that any admissible schedule can
  reach?
* **(Q3) Price.** What does the zero-over-trust point of that frontier cost, in the units of the
  estimand, and how does the cost scale with $K$ and with the tail of the residual law?
* **(Q4) Hardening.** What does the frontier say about a schedule that must survive contaminating
  residuals?

Write $y_{1:K}$ for a schedule and $m_k=\min_{i\le k}\hat\theta_i$ for the running minimum of the
estimates. Two quantities are compared throughout:

$$
\textsf{bias}(y):=\theta^\star_K-\mathbb E\,y_K,
\qquad
\textsf{overtrust}(y):=\mathbb E\max_{j\le K}\big(y_j-\hat\theta_j\big)^{+}. \tag{1.3}
$$

$\textsf{bias}$ is measured against the target that the threshold should have at step $K$.
$\textsf{overtrust}$ is the amount by which the schedule pushes the threshold above what the
residuals licensed: the total breach of the licence constraint, not of the descent certificate.
The two refer to different objects. Bias is a property of the *value* delivered at the final step;
over-trust is a property of the *whole path* relative to the licence.

---

## 2. The certificate cone

**Definition 2.1.** A schedule $y_{1:K}$ is *causal* if $y_k$ is a function of
$\hat\theta_{1:k}$ alone. It is *admissible* if it is causal and non-increasing. The
**certificate cone** is

$$
\mathcal C(\hat\theta)=\big\{y:\ y\ \text{causal, non-increasing},\ y_k\le\hat\theta_k\ \text{for all }k\big\}.
$$

Non-increasing plus $y_k\le\hat\theta_k$ is exactly $\textsf{overtrust}(y)=0$; causality is the
requirement that the schedule can be computed online.

**Proposition 2.2 (greatest element).** $\mathcal C(\hat\theta)$ is non-empty and has a pointwise
greatest element, namely the running minimum $y_k=m_k=\min_{i\le k}\hat\theta_i$.

*Proof.* Positivity: $m$ is causal, $m_k\le m_{k-1}$, and $m_k\le\hat\theta_k$, so $m\in\mathcal C$.
Maximality: for any $y\in\mathcal C$ and any $k$, monotonicity gives $y_k\le y_j$ and
admissibility gives $y_j\le\hat\theta_j$ for every $j\le k$; hence $y_k\le\min_{j\le k}\hat\theta_j=m_k$. $\square$

Two readings of Proposition 2.2 are worth keeping apart, because the literature conflates them.

**Corollary 2.3 (it is not a projection).** The running minimum is not the $L_p$ isotonic
regression of $\hat\theta$ onto the non-increasing cone for any $p<\infty$, and it is not the
$L_\infty$ isotonic regression either.

*Proof.* Take $\hat\theta=(0,3,1,2)$. The running minimum is $(0,0,0,0)$; the $L_2$ isotonic fit is
$(1.5,1.5,1.5,1.5)$; the $L_\infty$ isotonic fit is $(1.5,1.5,0.5,0.5)$. All three differ. $\square$

The gap is not a small numerical difference. Over 800 independent sequences the $\ell_\infty$
error of the running minimum is exactly twice the $\ell_\infty$ error of the isotonic fit:

**Proposition 2.4 (the factor is exactly two).** Write $D=\max_k(\hat\theta_k-m_k)$ for the maximum
drawdown. Then
$$
\max_k\big|\hat\theta_k-m_k\big|=D,\qquad
\max_k\big|\hat\theta_k-y^{\mathrm{iso},\infty}_k\big|=\tfrac D2 .
$$

*Proof.* The first equality is immediate because $m_k\le\hat\theta_k$. For the second, write
$y^\star$ for the minimiser of $\max_k|\hat\theta_k-y_k|$ over non-increasing $y$ and let $c$ be
the optimal value.

*$c\le D/2$:* set $y_k=m_k+D/2$. This is non-increasing, and
$-D/2\le\hat\theta_k-y_k=\hat\theta_k-m_k-D/2\le D/2$ because $0\le\hat\theta_k-m_k\le D$.

*$c\ge D/2$:* let $k^\ast=\arg\max_k(\hat\theta_k-m_k)$ and let $j$ be an index attaining
$m_{k^\ast}$, so $\hat\theta_{k^\ast}-\hat\theta_j=D$ with $j\le k^\ast$. Any feasible $y$ is
non-increasing, hence $y_{k^\ast}\le y_j$, and
$\hat\theta_{k^\ast}\le y_{k^\ast}+c\le y_j+c\le\hat\theta_j+2c=\hat\theta_{k^\ast}-D+2c$,
giving $D\le 2c$. $\square$

*Verified*: the linear programme for the $L_\infty$ isotonic fit returns $D/2$ to $2.2\times10^{-16}$
on 800/800 random sequences of length 3–25.

The content of 2.2–2.4 is a design decision, not an approximation error. The published rule
maximises *domination* of the data, not *closeness* to it. It gives away a factor of two in
$\ell_\infty$ accuracy in exchange for the property that no step ever breaches the licence
constraint. That exchange is the subject of §3.

**Corollary 2.5 (optimality under every increasing objective).** Since $m$ dominates every
admissible schedule pointwise (Proposition 2.2), it is simultaneously optimal over
$\mathcal C(\hat\theta)$ for every objective that is increasing in the schedule. In particular it
is the **largest admissible schedule**, and therefore the one that truncates the fewest
observations: a larger threshold retains more entries. It is not optimal over schedules that
breach the licence (Theorem 3.2 exhibits strictly better ones), so this statement is about
$\mathcal C$, not about all schedules.

The word "conservative" is avoided here because it points in two directions: $m$ is the least
conservative schedule if conservatism means *distrusting the data*, and the most conservative if
it means *keeping the threshold low*. The unambiguous statement is the one above.

**Corollary 2.6 (uniqueness at zero over-trust).** Every $y\in\mathcal C$ satisfies
$y_K\le m_K$ pathwise, hence $\textsf{bias}(y)\ge b_K:=-\mathbb E\,m_K$, with equality for
$y=m$. Among zero-over-trust schedules the running minimum is the unique pointwise maximal one and
the unique bias minimiser *up to the final index*; schedules that also attain $y_K=m_K$ may differ
at earlier indices.

So the rule is not arbitrary, but it is also not forced by the descent certificate alone: the
certificate admits all of $\mathcal C$, and $\mathcal C$ is a continuum. §3 identifies what that
continuum costs.

### 2.1 The licence cannot be relaxed at the last step alone

A natural attempt to enlarge the admissible set is to drop the constraint $y_k\le\hat\theta_k$ at
early steps and keep it only at the end, on the grounds that earlier thresholds are overwritten and
only $y_K$ is delivered. **This changes nothing.**

**Proposition 2.7 (the terminal licence is implied).** Let $y$ be causal and non-increasing. Then
$y_K\le\hat\theta_K$ holds pathwise. Consequently the set
$\{y\ \text{causal, non-increasing},\ y_K\le\hat\theta_K\}$ equals $\mathcal C(\hat\theta)$.

*Proof.* By Proposition 2.2, $y_K\le m_K=\min_{i\le K}\hat\theta_i\le\hat\theta_K$. $\square$

The consequence is worth stating plainly, because it forecloses an otherwise attractive route.
Any schedule that ends above $m_K$ must either be non-monotone or use information beyond
$\hat\theta_{1:K}$: the terminal index is the last place where the licence can bind, not the first
place where it can be dropped. Enlarging the class therefore requires giving up monotonicity, not
giving up the licence.

This matters for what follows. Within $\mathcal C$ the running minimum is optimal in the strong
sense of Corollary 2.5. Outside $\mathcal C$, the classical estimators win on accuracy by orders of
magnitude, but they are not schedules: an unbiased mean of $\{\hat\theta_k\}$ exceeds $m_K$ on the
overwhelming majority of paths, and hence breaches the licence at every index where the running
minimum was attained. Numerical check at $K=50$ over $2\times10^5$ paths: $\bar\theta$ exceeds
$m_K$ on $97.9\%$ of paths, and a candidate that replaces only the last value of $m$ by
$\min(\bar\theta,\hat\theta_K)$ is still inadmissible.

---

## 3. The frontier

Throughout this section $\hat\theta_{1:K}$ is an arbitrary random vector with $\mathbb E|m_K|<\infty$.
No independence, no Gaussianity, no model for $\hat\theta$ is used. Schedules are assumed integrable;
otherwise (3.2) holds trivially in the extended reals. For a general (non-stationary) estimate
sequence the target in (1.3) is read as $\theta^\star_K=\mathbb E\hat\theta_K$, so that Theorem 3.1
needs no stationarity either.

**Theorem 3.1 (frontier inequality).** For every non-increasing schedule $y$ and every $k$,

$$
y_k\ \le\ m_k+\max_{j\le k}\big(y_j-\hat\theta_j\big)^{+}. \tag{3.1}
$$

In particular $y_K\le m_K+V_K(y)$ with $V_K(y)=\max_{j\le K}(y_j-\hat\theta_j)^+$, and taking
expectations,

$$
\textsf{bias}(y)+\textsf{overtrust}(y)\ \ge\ b_K . \tag{3.2}
$$

*Proof.* Fix $k$ and let $j^\ast\in\arg\min_{j\le k}\hat\theta_j$, so $\hat\theta_{j^\ast}=m_k$. By
monotonicity $y_k\le y_{j^\ast}$, and by definition of the positive part
$y_{j^\ast}\le \hat\theta_{j^\ast}+(y_{j^\ast}-\hat\theta_{j^\ast})^+\le m_k+\max_{j\le k}(y_j-\hat\theta_j)^+$.
Taking $k=K$ and expectations gives (3.2). $\square$

Monotonicity is not decoration. A single example shows how (3.1) fails without it: take $K=2$,
$\hat\theta=(0.2,\,3)$ and the schedule $y=(0.1,\,0.5)$, feasible in every respect except
monotonicity. Then $m_2=0.2$, $V(y)=0$, and $y_2=0.5>m_2+V(y)$, so (3.1) fails. Sampling confirms
this is generic rather than pathological: 30000 randomly perturbed non-monotone schedules violated
(3.1) in 22.3% of cases, against zero of 30000 monotone ones.

**Theorem 3.2 (the frontier is attained, exactly).** For $\delta\ge0$ let

$$
y^{(\delta)}_k=\min_{i\le k}\hat\theta_i+\delta . \tag{3.3}
$$

Then $y^{(\delta)}$ is causal, non-increasing, and

$$
y^{(\delta)}_K=m_K+\delta,\qquad V_K\big(y^{(\delta)}\big)=\delta
\quad\text{for every realisation}. \tag{3.4}
$$

Hence $\textsf{bias}\big(y^{(\delta)}\big)=b_K-\delta$ and $\textsf{overtrust}\big(y^{(\delta)}\big)=\delta$:
the pair $(b_K-\delta,\delta)$ lies exactly on the line (3.2).

*Proof.* Causality and monotonicity are immediate from the running minimum, and
$y^{(\delta)}_K=m_K+\delta$ holds by definition. For $V_K$: at every index $j$,
$y^{(\delta)}_j-\hat\theta_j=\delta+(m_j-\hat\theta_j)\le\delta$ because $m_j\le\hat\theta_j$.
Equality is attained: let $j^\ast\in\arg\min_{i\le K}\hat\theta_i$; then $m_{j^\ast}=\hat\theta_{j^\ast}$,
so $y^{(\delta)}_{j^\ast}-\hat\theta_{j^\ast}=\delta$. Hence the maximum over $j\le K$ is exactly
$\delta$. $\square$

Equivalently, clipping the running minimum from below at any level $L\ge m_K$ also attains
(3.2), and the two constructions coincide on the terminal index. The schedule (3.3) is the one
that can be computed online.

**Corollary 3.3 (exchange rate one to one).** The map $\delta\mapsto(b_K-\delta,\delta)$
parametrises the reachable frontier exactly: one unit of over-trust buys one unit of bias, and
no schedule does better.

*Verified* ($K=40$, 20000 paths, same ensemble):

| $\delta$ | bias | bias drop | over-trust | bias $+$ over-trust |
|---|---|---|---|---|
| 0.00 | 2.1644 | 0.0000 | 0.0000 | 2.1644 |
| 0.10 | 2.0644 | 0.1000 | 0.1000 | 2.1644 |
| 0.50 | 1.6644 | 0.5000 | 0.5000 | 2.1644 |
| 1.00 | 1.1644 | 1.0000 | 1.0000 | 2.1644 |
| 2.00 | 0.1644 | 2.0000 | 2.0000 | 2.1644 |
| 3.00 | $-0.8356$ | 3.0000 | 3.0000 | 2.1644 |

The sum is constant to four decimals across a range of $\delta$ wide enough to make the bias
change sign. This is not an asymptotic statement and not an average: (3.4) holds pathwise.

**Corollary 3.4 (the isotonic fits do not minimise this functional).** Both isotonic fits lie
strictly above the line (3.2). The ensemble throughout is $\hat\theta_i$ i.i.d. $N(0,1)$ (the
distribution-free Theorem 3.1 does not need this; the numbers below do). With $K=10,40,150$ (2000
paths each):

| $K$ | $b_K$ | min: bias / over-trust / excess | $L_\infty$ iso: bias / over-trust / excess | $L_2$ iso: bias / over-trust / excess |
|---|---|---|---|---|
| 10 | 1.5415 | 1.5415 / 0.0000 / — | 1.3752 / 1.3709 / **+1.2046** | 0.5365 / 1.3881 / **+0.3832** |
| 40 | 2.1646 | 2.1646 / 0.0000 / — | 2.0027 / 2.0440 / **+1.8821** | 0.5665 / 2.1030 / **+0.5050** |
| 150 | 2.6464 | 2.6464 / 0.0000 / — | 2.5455 / 2.5504 / **+2.4495** | 0.6017 / 2.6150 / **+0.5702** |

What this shows is exactly what it says: the isotonic fits do not minimise the linear functional
$\textsf{bias}+\textsf{overtrust}$. It does **not** show that they are worse threshold estimators.
They are not. The $L_2$ isotonic fit cuts the terminal bias roughly threefold at $K=150$, and any
of these rules beats the running minimum by orders of magnitude in mean squared error — an
unbiased mean of the estimates beats it by a factor between $3\times10^{3}$ and $10^{5}$ over the
range of $K$ we simulated. The comparison of this corollary is a comparison within a constrained
class, and the constraint, not the estimator, is what separates the entries in the table.

One feature of the comparison favours the competitors and should be stated: the isotonic fits are
non-causal, pooling future estimates, and are granted this hindsight freely. That is offered as
context, not as an argument. It does not follow, and we do not claim, that a causal rule aiming at
closeness must do worse: causality is one restriction and monotonicity another, and the numbers in
the table are not a ranking of the two.

**Interpretation.** Over-trust is the schedule placing the threshold above the value the residuals
licensed. Theorem 3.1 says over-trust is the only currency in which bias can be bought, and
Corollary 3.3 says the exchange rate is exactly one. The zero-bias schedule is the one that already
knows the answer, and its cost is $b_K$ of over-trust.

---

## 4. The price

Theorem 3.2 leaves the number $b_K=-\mathbb E m_K$ to be computed. This section computes it in
three regimes and for the moving-target case.

### 4.1 Exact form

**Theorem 4.1.** Let $\hat\theta_i=\theta^\star+\varepsilon_i$ with $\varepsilon_i$ i.i.d.,
continuous, symmetric about zero, with cdf $F$. Then

$$
b_K=\int_0^\infty\Big[1-F(t)^K-\big(1-F(t)\big)^K\Big]\,dt , \tag{4.1}
$$

and for Gaussian $\varepsilon$ this is $b_K=\int_0^\infty[1-\Phi(t)^K-(1-\Phi(t))^K]dt$.

*Proof.* For a real random variable $Y$ with $\mathbb E|Y|<\infty$,
$\mathbb E Y=\int_0^\infty\mathbb P(Y>t)dt-\int_0^\infty\mathbb P(Y<-t)dt$. Apply this to
$Y=-m_K$. By independence and symmetry,
$\mathbb P(-m_K>t)=\mathbb P(m_K<-t)=1-\mathbb P(\hat\theta_i\ge-t\ \forall i)=1-F(t)^K$,
and $\mathbb P(-m_K<-t)=\mathbb P(m_K>t)=\mathbb P(\hat\theta_i>t\ \forall i)=(1-F(t))^K$.
$\square$

The integrand is bounded by $1/2$ at $t=0$ and decays like $K(1-F(t))$, so (4.1) is a
one-dimensional quadrature that costs nothing to evaluate. *Verified*: (4.1) agrees with
$10^4$–$2\times10^6$ Monte-Carlo paths to within $0.005$ for every
$K\in\{2,3,5,10,32,100,316,1000,3162,10^4,3\cdot10^4,10^5\}$.

**Proposition 4.2 (the floor).** For Gaussian $\varepsilon$, $b_2=1/\sqrt{\pi}=0.56418958\ldots$

*Proof.* At $K=2$, $1-F^2-(1-F)^2=2F(1-F)$; writing $F(t)=\Phi(t)$ and $1-\Phi(t)=\Phi(-t)$,
the integrand is $2[\Phi(-t)-\Phi(-t)^2]$, and
$\int_0^\infty 2[\Phi(-t)-\Phi(-t)^2]dt=\mathbb E[\max(\varepsilon_1,\varepsilon_2)]
=\mathbb E[\frac12(\varepsilon_1+\varepsilon_2+|\varepsilon_1-\varepsilon_2|)]
=\frac12\mathbb E|N(0,2)|=1/\sqrt\pi$. The step $\int_0^\infty 2[\Phi(-t)-\Phi(-t)^2]dt
=\mathbb E\max(\varepsilon_1,\varepsilon_2)$ uses $\mathbb E\max=\int_0^\infty\mathbb P(\max>t)dt$
together with $\mathbb P(\max\le -t)=\Phi(-t)^2$. $\square$

*Verified*: the quadrature returns $0.56418958$ and $4\times10^6$ paths return $0.5640$ (s.e. $0.0004$).

### 4.2 Growth in $K$

Two different objects are compared here, and only one of them is unusable. The bare asymptotic
$\sqrt{2\ln K}$ overstates the price by $40\%$ at $K=10$ and by $9\%$ at $K=10^5$; it should not be
quoted. The corrected expansion $b^{\rm c}_K+\gamma/b^{\rm c}_K$ (Euler term included) is a usable
closed form: it is within $6.0\%$ at $K=10$, $1.95\%$ at $K=100$, $0.94\%$ at $K=10^3$, $0.55\%$ at
$K=10^4$ and $0.37\%$ at $K=10^5$, monotonically improving. From $K\gtrsim10^3$ it can replace the
quadrature where a quadrature is inconvenient, at a cost under $1\%$. The exact statement, when
accuracy matters, is still (4.1).

| $K$ | exact (4.1) | $\sqrt{2\ln K}$ | centring $b^{\rm c}_K$ | $b^{\rm c}_K+\gamma/b^{\rm c}_K$ | error of the last |
|---|---|---|---|---|---|
| 10 | 1.5388 | 2.1460 | 1.3619 | 1.6309 | $+6.0\%$ |
| 100 | 2.5076 | 3.0349 | 2.3663 | 2.5565 | $+1.95\%$ |
| $10^3$ | 3.2414 | 3.7169 | 3.1165 | 3.2718 | $+0.94\%$ |
| $10^4$ | 3.8516 | 4.2919 | 3.7384 | 3.8729 | $+0.55\%$ |
| $10^5$ | 4.3843 | 4.7985 | 4.2802 | 4.4005 | $+0.37\%$ |

The rate is what is robust: $b_K/\sqrt{2\ln K}\to1$.

### 4.3 Heavy tails

The Gaussian price is the smallest of the three regimes. For unit-variance $t_\nu$ the lower tail
is $F(-x)\sim C_\nu x^{-\nu}$, and the minimum becomes a Fréchet-type functional with an algebraic
price:

**Proposition 4.3.** For symmetric $\varepsilon$ with $F(-x)\sim C_\nu x^{-\nu}$ and $\nu>1$,
$$
b_K\sim (KC_\nu)^{1/\nu}\,\Gamma\!\Big(1-\frac1\nu\Big)\qquad(K\to\infty), \tag{4.2}
$$
so the price grows like $K^{1/\nu}$ rather than like $\sqrt{\ln K}$.

*Proof.* $b_K=\mathbb E[-m_K]=\int_0^\infty\mathbb P(-m_K>t)\,dt=\int_0^\infty[1-(1-F(-t))^K]dt$.
Substituting $u=KC_\nu t^{-\nu}$ gives $dt=-\frac1\nu(KC_\nu)^{1/\nu}u^{-1/\nu-1}du$ and the
integrand $1-(1-u/K)^K\to 1-e^{-u}$, so the integral converges to
$(KC_\nu)^{1/\nu}\frac1\nu\int_0^\infty(1-e^{-u})u^{-1/\nu-1}du$. The remaining integral is
$\int_0^\infty u^{-1/\nu-1}(1-e^{-u})du=\nu\,\Gamma(1-1/\nu)$ for $\nu>1$ — note the factor $\nu$,
which cancels the $1/\nu$ — so the limit is $(KC_\nu)^{1/\nu}\Gamma(1-1/\nu)$. Dominated
convergence justifies passing the limit inside since $1-(1-u/K)^K\le\min(1,u)$. $\square$

*Remark.* The cancellation is worth recording because it is easy to lose: the substitution
contributes $1/\nu$ and the Gamma integral contributes $\nu$. Writing the substitution without
cancelling gives an expression a factor $\nu$ too small, which is an error of $3$–$6$ in the
constant at the tail indices considered here.

*Verified* ($C_\nu$ folded in; exact column by quadrature to $10^{-11}$, simulation column with
$2\times10^5$ paths at $K\le10^3$ and $10^5$ at $K=10^4$; figures in parentheses are
simulated/exact):

| $\nu$ | $K=10^2$ | $K=10^3$ | $K=10^4$ | log-log slope | $1/\nu$ | exact/(4.2) at $K=10^4$ |
|---|---|---|---|---|---|---|
| 3 | 3.6164 (3.6219) | 8.0106 (8.0176) | 17.362 (17.374) | 0.341 | 0.333 | 0.998 |
| 4 | 3.3365 (3.3439) | 6.2665 (6.2675) | 11.364 (11.323) | 0.263 | 0.250 | 0.993 |
| 6 | 3.0214 (3.0208) | 4.8765 (4.8788) | 7.4468 (7.4487) | 0.196 | 0.167 | 0.969 |

The simulation/exact ratios are $0.997$–$1.004$ with standard errors $0.001$–$0.002$; the single
outlier is $\nu=4$, $K=10^4$ ($+2.6$ s.e.), consistent with the $t_4$ sampling variance. Slopes are
fitted on the last three decades.

This quantifies a qualitative complaint that already appears in the capped-Frobenius paper:
heavy-tailed residuals make the threshold deteriorate faster than the Gaussian heuristic predicts.
Proposition 4.3 says how much faster: at $\nu=3$ and $K=10^4$ the price is $17.37$ against a
Gaussian $3.85$, a factor $4.5$.

Unlike the Gaussian case of §4.2, here the asymptotic constant is accurate at usable lengths. The
ratio exact/(4.2) runs $0.966,0.993,0.998,1.000$ for $\nu=3$ at $K=10^2,\dots,10^5$; $0.927,0.977,
0.993,0.998$ for $\nu=4$; and $0.846,0.931,0.969,0.986$ for $\nu=6$. So (4.2) is within $3.1\%$ for
every $\nu$ once $K\ge10^4$, within $2.3\%$ for $\nu\le4$ from $K\ge10^3$, but understates by up to
$15\%$ at $\nu=6$, $K=10^2$. The practical price remains the quadrature (4.1) with the tail of $F$
inserted; the asymptotic form is a rate, and at small $\nu$ it is also a good number.

### 4.4 Moving target

The price $b_K$ is for a stationary target. When the target itself decreases, the annealed
minimum stops looking at early steps and the bias falls. The decline is no longer $\sqrt{2\ln K}$;
it is governed by how many steps lie within the terminal band.

**Proposition 4.4 (lower bound).** For any target sequence $\theta^\star_{1:K}$,

$$
\textsf{bias}\ \ge\ \max_{\delta\ge0}\Big[\beta_{n(\delta)}-\delta\Big],
\qquad
n(\delta)=\#\{i\le K:\ \theta^\star_i-\theta^\star_K\le\delta\big\}, \tag{4.3}
$$

where $\beta_n$ is the price of an $n$-step block with the same residual law.

*Proof.* Fix $\delta\ge0$ and let $I=\{i:\theta^\star_i-\theta^\star_K\le\delta\}$, $n=|I|$. For a
non-increasing target the set $I$ is the terminal block $\{K-n+1,\dots,K\}$, so its indices are
consecutive and the minimum over $I$ is a minimum of $n$ i.i.d. errors around a target that does
not vary by more than $\delta$. Two elementary steps:

(i) $m_K\le\min_{i\in I}\hat\theta_i$, because the running minimum over all of $\{1..K\}$ is no
larger than the minimum over a subset.

(ii) For every $i\in I$, $\hat\theta_i=\theta^\star_i+\varepsilon_i\le\theta^\star_K+\delta+\varepsilon_i$,
hence $\min_{i\in I}\hat\theta_i\le\theta^\star_K+\delta+\min_{i\in I}\varepsilon_i$.

Together, $\mathbb E[m_K]\le\theta^\star_K+\delta+\mathbb E\min_{i\in I}\varepsilon_i
=\theta^\star_K+\delta-\beta_n$, using that $\varepsilon_i$ are i.i.d. so
$\mathbb E\min_{i\in I}\varepsilon_i=-\beta_n$. Rearranging gives
$\textsf{bias}=\theta^\star_K-\mathbb E[m_K]\ge\beta_n-\delta$. Maximising over $\delta$ gives
(4.3). $\square$

The bound is valid but not tight: the simulated bias sits between $\beta_{n(\delta)}$ and
$\beta_{K}$. It is superseded by the exact law below.

**Theorem 4.5 (exact law for a deterministic moving target).** Let
$\hat\theta_i=\theta^\star_i+\varepsilon_i$ with $\varepsilon_i$ i.i.d. with continuous cdf $F$, and
let $a_i=\theta^\star_i-\theta^\star_K\ge0$ be the deterministic offsets. Put
$Z=\min_{i\le K}(\varepsilon_i+a_i)$. Then

$$
\textsf{bias}\;=\;\int_0^\infty\Big[1-\prod_{i=1}^K\big(1-F(-t-a_i)\big)\Big]dt
\;-\;\int_0^\infty\prod_{i=1}^K\big(1-F(t-a_i)\big)\,dt . \tag{4.4}
$$

*Proof.* $Z>t$ if and only if $\varepsilon_i>t-a_i$ for every $i$, so by independence
$\mathbb P(Z>t)=\prod_i(1-F(t-a_i))$. Likewise $\mathbb P(Z<-t)=1-\prod_i(1-F(-t-a_i))$. Apply
$\mathbb E Z=\int_0^\infty[\mathbb P(Z>t)-\mathbb P(Z<-t)]dt$ and negate. $\square$

No symmetry, stationarity or Gaussianity is assumed. Setting $a_i\equiv0$ and $F$ symmetric
recovers (4.1): the first term becomes $1-\Phi(t)^K$ and the second $\Phi(-t)^K=(1-\Phi(t))^K$. The
product costs $O(K)$ per quadrature node and is stable in log-space, so (4.4) is as cheap as (4.1).

*Verified* ($K=1000$, linear drift of total drop $D$, $4\times10^5$ paths per row; "bound" is the
best case of (4.3)):

| drop $D$ | exact (4.4) | simulated | s.e. | bound (4.3) | binding $n$ |
|---|---|---|---|---|---|
| 0 | 3.2414 | 3.2404 | 0.0006 | 3.241 | 1000 |
| 2 | 2.6642 | 2.6635 | 0.0006 | 2.353 | 166 |
| 8 | 2.2082 | 2.2074 | 0.0007 | 1.856 | 46 |
| 16 | 1.9577 | 1.9565 | 0.0007 | 1.579 | 24 |
| 64 | 1.4042 | 1.4035 | 0.0009 | 0.942 | 9 |
| 512 | 0.4843 | 0.4857 | 0.0012 | 0.000 | 1 |

Every row agrees to within $1.9$ standard errors; the largest absolute discrepancy is $0.0014$. The
gap between the bound (4.3) and the truth is entirely accounted for by indices lying just outside
the terminal band, which (4.3) discards and (4.4) prices.

**Remark 4.6 (scope).** (4.4) requires the drift to be deterministic and the errors i.i.d. If the
drift is random but independent of the errors, (4.4) is a *conditional* exact law and the
unconditional bias is its expectation over the drift path. If the two are dependent, for instance
when the contamination rate and the residual scale move together, (4.4) does not apply and
Proposition 4.4 remains the available statement.

**Remark 4.7 (where the minimum lives).** The binding index count of Proposition 4.4 is predicted by
a one-line scaling argument. For linear drift with per-step slope $D/K$, matching the two terms of
$\beta_{n}-\delta$ gives

$$
n^\ast\sqrt{2\ln n^\ast}=\frac{K}{D},\qquad \delta^\ast=\frac{n^\ast D}{K}, \tag{4.5}
$$

yielding $n^\ast=157.2,\,45.3,\,24.7,\,7.7$ for $D=2,8,16,64$ at $K=1000$, against the empirically
binding values $166,46,24,9$. The band estimate $\beta_{n^\ast}-\delta^\ast$ recovers (4.3); the
residual between it and (4.4) is the price of the transition region. A limit theorem for (4.4) as
$K\to\infty$ with the drift rate fixed, that is, the law of $\min_{j\ge1}(\varepsilon_j+sj)$ after
centring, is the remaining asymptotic question (§8).

---

## 5. Breakdown, and what the frontier permits

Theorem 3.1 bounds the bias from below; it does not bound it from above. In fact the
zero-over-trust point of the frontier has the worst possible upper bound.

**Proposition 5.1 (no uniform positive breakdown level).** Suppose each estimate is displaced
downward by at least $\Delta$ independently with probability $\varepsilon$. Then with probability
$1-(1-\varepsilon)^K$ at least one displacement occurs, the running minimum carries it, and every
later threshold inherits it. For any fixed $\varepsilon>0$ this probability tends to one as
$K\to\infty$; hence the rule has no breakdown level $\varepsilon^\ast>0$ that is uniform in the
block length.

*Terminology.* This is the block-length statement, not the classical finite-sample breakdown point
of an estimator, and it should not be read as one. The classical notion asks for the largest
fraction of contaminated observations below which an estimator is guaranteed to stay bounded; the
statement here says that under element-wise contamination of the estimate sequence the probability
of *some* contamination in the block grows to one, so no such fraction exists. The two questions
are related but the numbers are not interchangeable.

*Scope.* The statement is about element-wise independent contamination of the estimate sequence.
It does not describe structured contamination, where one bad element is shared across many
thresholds, and it makes no claim about reconstruction quality; §7.5 is where that second
distinction is drawn.

*Verified* ($K=200$, $\Delta=8$, flat target, 20000 paths):

| $\varepsilon$ | $\mathbb P(\text{outlier in block})$ | min rule, mean $m_K$ | min rule conditioned on poison |
|---|---|---|---|
| 0 | 0.000 | $-2.75$ | — |
| 0.002 | 0.330 | $-4.50$ | $-8.11$ |
| 0.010 | 0.866 | $-7.74$ | $-8.53$ |
| 0.050 | 1.000 | $-9.51$ | $-9.51$ |

The mean bias moves from $2.75$ to $9.51$ as $\varepsilon$ goes from $0$ to $5\%$. The running
minimum has no forgetting mechanism.

### 5.1 The threshold sequence as a robustness path

The results above are all about the sequence. It is worth saying what that sequence is, in the
language of robust estimation, because the connection is exact and it costs one paragraph.

For a redescending loss with threshold $t$, the rejection point is $t$ and the gross-error
sensitivity is $\gamma^\ast=\sup_r|\psi(r/t)|$. For the capped and half-quadratic families
$\gamma^\ast=1$ and the rejection point is $t$ itself; for Welsch and Cauchy the rejection point is
infinite and $\gamma^\ast$ is $0.607$ and $0.500$ respectively. The first fact that matters is that
for the family used in all five methods, **the threshold *is* the rejection point**.

A non-increasing threshold sequence is therefore a path along which the rejection point shrinks
monotonically. Measured over $K=200$ steps of a unit-noise experiment:

| schedule | terminal $y_K$ | $\sup_k y_k$ (the largest rejection point reached) |
|---|---|---|
| constant $y=3$ | 3.000 | 3.000 |
| geometric $y_k=3\cdot0.98^k$ | 0.054 | 3.000 |
| running minimum | 0.009 | 1.188 |

The running minimum is not merely monotone; it is the schedule with the smallest largest rejection
point among the three, because it never rises above the data. In the influence-function sense it is
the **most conservative** schedule, not the least — which is the opposite of what "least
conservative" would suggest, and is why Corollary 2.5 states its claim in terms of size rather than
of conservatism.

The second fact is that the contamination fraction lives on a different axis from the classical one.
Classical robustness asks what fraction of *observations* may be corrupted before an estimator
fails; here the relevant fraction is the fraction of *steps* that carry a displaced estimate, and
for a block of length $K$ the probability of at least one such step is $1-(1-\varepsilon)^K$. At
$\varepsilon=0.2\%$ per step and $K=200$ that probability is already $33\%$. A statement such as
"the scale estimator has a $50\%$ breakdown point" is therefore not the right defence for a
threshold sequence; what matters is the length of the block and the per-step contamination rate
separately.

What the classical theory does not price, and what §3–§4 do, is the *cost* of this path: how far
below the unbiased threshold it ends, and what it would cost to stop there.

**Proposition 5.2 (hardening is a frontier move).** Fix $\ell\ge0$ and let
$y_k=\max\{\min_{i\le k}\hat\theta_i,\ -\ell\}$. Then $y$ is causal and non-increasing, so
Theorem 3.1 applies to it, and

$$
\textsf{bias}(y)\ \le\ \min\{b_K,\ \ell\},\qquad
\textsf{overtrust}(y)\ \ge\ b_K-\min\{b_K,\ell\}. \tag{5.1}
$$

In the adversarial limit (every step poisoned) the bias is pinned at $\ell$ instead of growing with
$\Delta$.

*Proof.* Monotone because the running minimum is monotone and $\max\{\cdot,-\ell\}$ is monotone;
causal by construction. For the bias: $y_K\ge m_K$ gives $\textsf{bias}\le b_K$, and $y_K\ge-\ell$
gives $\textsf{bias}\le\ell$. The over-trust bound is (3.2) rearranged. $\square$

*Verified* ($K=200$, $\Delta=8$, floor at $-\ell=-b_{200}=-2.746$, $2\times10^4$ paths; the floor is
$b_K$ from the quadrature (4.1), not the asymptotic centring constant):

| $\varepsilon$ | min rule | floored rule | floored bias | $b_K$ | over-trust of floored rule |
|---|---|---|---|---|---|
| 0 | $-2.7431$ | $-2.5880$ | 2.5880 | 2.7460 | 0.1551 |
| 0.002 | $-4.4992$ | $-2.6381$ | 2.6381 | 2.7460 | 1.8611 |
| 0.010 | $-7.7844$ | $-2.7255$ | 2.7255 | 2.7460 | 5.0590 |
| 0.050 | $-9.4970$ | $-2.7460$ | 2.7460 | 2.7460 | 6.7510 |

At $\varepsilon=0$ the floor is inactive on average, so the floored bias stays below $b_K$
($2.588$ against $2.746$) and the over-trust is small; as contamination grows the floor binds on
more paths, the bias rises monotonically to $\ell$, and the over-trust absorbs the difference. Both
limits of (5.1) are visible in the table: the bias is bounded by $\min\{b_K,\ell\}$ throughout, and
the over-trust is bounded below by $b_K-\textsf{bias}$, with equality only in the limit
$\varepsilon\to1$.

**Where the optimum sits.** With $K=50$, $\Delta=8$, $b_{50}=2.249$ and offsets $\delta=c\,b_{50}$
(cells are root-mean-square error of the terminal threshold, against the true threshold at that
step):

| $\varepsilon$ | $\delta=0$ | $\delta=b_K/2$ | $\delta=b_K$ | $\delta=2b_K$ |
|---|---|---|---|---|
| 0 | 2.302 | 1.222 | **0.463** | 2.290 |
| 0.005 | 4.312 | 3.455 | 2.793 | **2.668** |
| 0.020 | 6.806 | 5.819 | 4.890 | **3.408** |
| 0.050 | 8.372 | 7.283 | 6.206 | **4.129** |

Without contamination the frontier's corner point $\delta=b_K$ is exactly optimal, and the
root-mean-square error collapses to the sampling standard deviation $0.463$. With contamination the
optimum moves outward and the bias is no longer the dominant term. This is the sense in which the
corner point is the right default and the wrong universal rule: it minimises bias, not risk.

The table is a statement about the *threshold*. It does not settle what a schedule with lower
threshold error reconstructs; §7.5 shows that the retained set, which the threshold does not fix,
moves the reconstruction independently.

---

## 6. Choosing $\delta$

Corollary 3.3 leaves the choice of $\delta$ open, and Theorem 3.1 says the choice is a
budget allocation. Three regimes, in the order a practitioner meets them.

**(a) Honest residuals, accuracy objective.** Within the one-parameter shift family
$y_k=m_k+\delta$, the mean-squared-error criterion is

$$
\mathbb E\big[\big(\theta^\star_K-y^{(\delta)}_K\big)^2\big]=(\theta^\star_K-\mathbb E m_K-\delta)^2+\operatorname{Var}(m_K),
$$

whose minimiser is $\delta=b_K$. Since $\operatorname{Var}(m_K)$ does not depend on $\delta$, the
minimiser is unique in the family and the bias-corrected schedule is unbiased. Two limits are worth
stating explicitly:

* (i) Unbiasedness holds in the i.i.d. symmetric model; if the estimates are correlated the required
  shift is smaller than $b_K$ and unknown (§8 item 2).
* (ii) The claim is **not** that $m_K+b_K$ is the best estimator of the threshold. It is the best
  within the shift family. Plain averaging of the estimates has a mean-squared error smaller by
  between $1$ and $4$ orders of magnitude over $K=10$ to $10^{3}$ (ratios $27$ and $1.1\times10^{4}$
  at the two ends), but it is not non-increasing and it breaches the licence wherever the running
  minimum was attained (§2.1).

The honest form of the claim is therefore conditional, and the condition is the subject of the
paper: among schedules that are causal and never breach the licence, the shift by $b_K$ is the
unbiased one, and it pays exactly $b_K$ of over-trust to get there (Theorem 3.2).

**(b) Honest residuals, robustness objective.** If the requirement is that no single bad step move
the threshold far, the floored rule of Proposition 5.2 is a feasible schedule under a user-chosen
hard cap on worst-case deviation; for a given floor $-\ell$ its worst-case bias is $\ell$ by
construction. It is not the efficient choice: §5 shows it sits strictly above the frontier relative
to the shift (3.3), because the floor binds only on the poisoned part of the path while (3.3)
applies the cheapest possible offset throughout. The only justification for the floor over the
shift is that the application can commit, in advance, to an absolute over-trust ceiling.

**(c) Contaminated residuals.** In the synthetic threshold-only criterion of §5, the optimum moves
from $\delta=b_K$ at $\varepsilon=0$ to $\delta=2b_K$ by $\varepsilon=0.5\%$ and stays there through
$\varepsilon=5\%$. The mechanism is transparent: a downward outlier of size $\Delta$ costs the min
rule $\Delta-b_K$ of bias but the shifted rule only $\max(0,\Delta-b_K-\delta)$, so $\delta$ should
be set at the size of the largest excursion the threshold statistic can produce, capped by the
tolerance.

The solver experiment of §7.4 finds the *sign* of the same effect — relaxation helps when the
residuals are clean and hurts when they are not — but over a different range of contamination and
against a different performance measure. The two are directionally consistent and not
quantitatively interchangeable: the threshold-only criterion has no retained-observation layer, so
it cannot price what §7.5 shows to be a separate effect.

The single sentence this section supports is that "$3\times$MADN" and "$3\sigma$" are not
calibration constants but an implicit budget of over-trust. The two move at different layers:
$\zeta$ rescales the *input* sequence, $\hat\theta_k=\zeta\,T(r_k)$, while $\delta$ translates the
*output* schedule, $y_k=m_k+\delta$. They are therefore not operationally interchangeable: changing
$\zeta$ changes what gets minimised, changing $\delta$ changes what is done with the minimiser, and
the frontier prices only the second. What the frontier makes precise is that both are denominated in
the same currency, one unit of over-trust per unit of bias.

---

## 7. Validation on a solver

Scope: this section establishes what the objects of §3–§5 are in an annealing loop driven by real
residuals, and where the theory stops. It does not compare against a benchmark.

**Setting.** Rank-3 matrix, $n=100$, observed on a fraction $p=0.6$ of the entries, Gaussian noise
with $\sigma=0.05$, a fraction $\varepsilon$ of the observed entries corrupted by outliers of size
$6\sigma\cdot\mathcal U[0.5,1.5]$. Loss: capped Frobenius, $\sum_{ij}\min(r_{ij}^2,\theta^2)$.
State update: the proximal block-coordinate step of the published algorithm, with the proximal term
retained; initialised from the rank-3 singular value decomposition of the Winsorised matrix (entries
clipped at $3\times$MADN before the decomposition, because an uncorrected decomposition follows the
outlier block, whose Frobenius norm exceeds the signal's). Threshold: $\zeta$ times the MADN with
$\zeta=3$, then the rule (1.1). $K=200$ iterations, 20 replications per condition.

### 7.1 The rule is reproduced exactly

The recorded schedule equals the running minimum of the recorded unconstrained path exactly, to
machine precision, for every replication and every $\varepsilon$. There is no free constant in this
check: the identification of (1.1) with the running minimum of $\{\hat\theta_k\}$ holds as an
identity. Reconstruction error is $2.77\%$ clean and $3.35\%$ at $\varepsilon=10\%$ (the same runs
that give the $\mathrm{rel}=0$ column of §7.4).

### 7.2 The price depends on the effective history, not the iteration count

The unconstrained path converges and then freezes. After 1–3 iterations the estimate stops moving:
the last fifty values have within-path standard deviation of order $10^{-14}$, and the argmin of the
path sits at index 0–3 out of 200. The block of $K=200$ estimates is therefore not a block of 200
independent re-estimations; it is a short transient followed by a constant, and the running minimum
is a transient value rather than an extremal fluctuation.

The consequence is quantitative. The price $b_n$ counts *independent* re-estimations, so the
operative $n$ is the length of the transient, not the iteration count. Three candidate definitions
of that length, recorded on the same runs ($20$ replications per contamination level):

| $\varepsilon$ | $n_{\rm total}=K$ | $n_{\rm trans}$ (within 1% of terminal) | $n_{\rm plateau}$ (last distinct value) |
|---|---|---|---|
| 0 | 200 | 1.05 | 12.8 |
| 0.02 | 200 | 2.00 | 14.2 |
| 0.05 | 200 | 2.20 | 15.1 |
| 0.10 | 200 | 2.65 | 16.6 |

$b_{200}=2.746$ against $b_{n_{\rm trans}}\in\{0,0.564,0.564,0.846\}$: the iteration count
overstates the price by a factor between $3$ and infinity, and the overstatement grows as the
solver converges faster.

A direct demonstration of what that mis-scaling does: taking $b_{200}$ as an absolute offset for the
truncation threshold puts the threshold an order of magnitude above its own value, at which point
truncation stops acting altogether and the schedule's $\delta$ ceases to matter. The scale of $b_K$
is the spread of an i.i.d. estimator, and in a converging solver that spread has gone to zero.

The two candidate definitions differ by a factor of about six, and which is right depends on what
one asks the price to predict. $n_{\rm trans}$ is the length over which the estimate is still moving
appreciably; $n_{\rm plateau}$ is the length over which it is still changing at all. Neither is
derived from the theory, and choosing between them is left open (§8 item 2) rather than settled by
the numbers above.

### 7.3 The ratchet is total

Monotonicity is one-way, and no non-increasing schedule can follow a rise:

| run | $\hat\theta$ before step 100 | $\hat\theta$ after | ratio | $\theta_K$ | follows? |
|---|---|---|---|---|---|
| 1 | 0.2908 | 0.7555 | 2.60 | 0.2908 | no |
| 2 | 0.2886 | 0.7650 | 2.65 | 0.2877 | no |
| 3 | 0.2860 | 0.7388 | 2.58 | 0.2858 | no |
| 4 | 0.2856 | 0.7490 | 2.62 | 0.2851 | no |
| 5 | 0.2786 | 0.7964 | 2.86 | 0.2781 | no |
| 6 | 0.2885 | 0.7528 | 2.61 | 0.2885 | no |

Noise was tripled at step 100. The estimate moves by a factor $2.6$; the schedule moves by $0$. The
threshold ends up a factor $2.6$ below what the current residuals license. This is not a defect of
the estimation — it is what "non-increasing" means — and it holds for every offset the estimator's
own excursion scale would justify: the frozen minimum stays binding for every $\delta\le0.45$,
the crossover being $\delta^\ast=\hat\theta_K-m_K\in[0.451,0.497]$ across the six runs.

A fixed positive offset below the current estimate is one way to retain adaptation when the scale
rises; it is not the only non-increasing schedule that could. What the argument does establish is
that no non-increasing schedule can *follow* an increase, so whatever is done must be decided in
advance, and its cost is priced by Theorem 3.2.

### 7.4 Relaxing the truncation

All rules share the threshold path and differ in how far below it the truncation sits. Writing
$\theta_{\rm trunc}=(1+\text{rel})\cdot\min_k\hat\theta_k$ and reporting the relative
reconstruction error:

| $\varepsilon$ | rel $=0$ | $5\%$ | $10\%$ | $20\%$ | $30\%$ |
|---|---|---|---|---|---|
| 0 | 0.0277 | 0.0276 | 0.0275 | 0.0274 | **0.0273** |
| 0.02 | 0.0296 | **0.0295** | 0.0296 | 0.0297 | 0.0299 |
| 0.05 | **0.0313** | 0.0314 | 0.0316 | 0.0323 | 0.0331 |
| 0.10 | **0.0335** | 0.0343 | 0.0350 | 0.0369 | 0.0395 |

Standard errors are $3\times10^{-4}$ to $5\times10^{-4}$. Paired differences against rel $=0$
(positive = relaxation better): $+0.00038$ ($t=+6.6$) at $\varepsilon=0$; $+0.00006$ ($t=+1.4$) at
$\varepsilon=0.02$; $-0.00177$ ($t=-9.6$) at $\varepsilon=0.05$; $-0.00598$ ($t=-20.7$) at
$\varepsilon=0.10$, all at rel $=30\%$.

The sign changes between the tested contamination levels. We do not extrapolate a critical rate
from four points, and the best cell in the clean row lies at the edge of the scanned grid, so it is
the best *within the grid* and not an interior optimum. What the table supports is narrower and
sufficient for the argument: relaxation is a real lever whose direction depends on whether
contamination or bias dominates.

### 7.5 What the frontier does not determine

The theory of §3–§5 is about the threshold sequence. It is silent about a second decision that the
threshold induces: *which* observations survive truncation. That decision is not a function of the
threshold alone, and its effect is not in the bias/over-trust accounting.

To isolate it we fix the threshold path and the initialisation, allow a common settling phase of 60
proximal sweeps, verify that the objective has flattened, and vary only the keep-mask. Under
row-structured contamination (whole rows displaced), four masks are compared: contamination-aware
deletion (drop exactly the corrupted entries), threshold-based retention (the published rule's
mask), full retention, and a row-wise count-matched random mask that keeps the same number of
entries per row as the published rule but chooses their positions uniformly.

| $\varepsilon$ | keep-all | threshold-kept | count-matched | contamination-aware |
|---|---|---|---|---|
| 0.05 | 0.1045 ± 0.0052 | 0.1041 ± 0.0051 | 0.1172 ± 0.0083 | **0.1036 ± 0.0050** |
| 0.10 | 0.1545 ± 0.0084 | 0.1531 ± 0.0083 | 0.1747 ± 0.0116 | **0.1510 ± 0.0082** |
| 0.20 | 0.2191 ± 0.0077 | 0.2175 ± 0.0075 | 0.2555 ± 0.0100 | **0.2138 ± 0.0072** |

Paired against threshold-kept: contamination-aware is better by $0.0006$–$0.0037$
($t=-1.5,-5.0,-5.4$), full retention worse by $0.0003$–$0.0016$ ($t=+1.2,+2.4,+4.0$), and the
count-matched mask worse by $0.0130$–$0.0380$ ($t=+2.5,+3.1,+5.7$). The ordering is the same at all
three contamination levels, and every difference is an order of magnitude smaller than the
contamination penalty itself ($0.05$ to $0.22$).

Two conclusions, both deliberately weak. First, at a fixed threshold the retained set remains an
additional degree of freedom, and it moves the reconstruction error by up to $0.038$ — an amount the
bias/over-trust pair cannot see. Second, the ordering is *not* a statement that correct truncation
is harmful: contamination-aware deletion, which knows exactly which entries are corrupted, is the
best of the four throughout. The count-matched arm is worse than threshold-based retention, but it
differs from it in both count and composition — it keeps corrupt entries as readily as clean ones —
so this experiment does not separate "how many" from "which ones", and we do not claim a universal
ordering of masks.

The scope statement this supports is the one that matters for the paper: a threshold-layer optimum
does not by itself determine a recovery-layer optimum, because the keep-mask is a further decision
that the theory does not price.

---

## 8. What remains open

1. **Moving target, asymptotic refinement.** The finite-sample problem is closed by Theorem 4.5.
   What remains is the asymptotic decomposition of (4.4) as $K\to\infty$ with the drift rate $D/K$
   fixed — the terminal band plus the transition region — equivalently the law of
   $\min_{j\ge1}(\varepsilon_j+sj)$ after centring. Remark 4.7 fixes the band; the transition
   region has no closed form.

2. **Effective number of re-estimations.** The price $b_n$ counts independent re-estimations. Two
   mechanisms make the operative $n$ smaller than $K$.

   *Autocorrelation.* With AR(1) residuals of correlation $\rho$ the price falls much more slowly
   than $K_{\rm eff}=K(1-\rho)/(1+\rho)$ suggests:

   | $K$ | $\rho$ | simulated bias | $b_K$ | $K_{\rm eff}$ | $b_{K_{\rm eff}}$ |
   |---|---|---|---|---|---|
   | 200 | 0.50 | 2.704 | 2.746 | 67 | 2.361 |
   | 200 | 0.90 | 2.337 | 2.746 | 10 | 1.586 |
   | 200 | 0.99 | 1.291 | 2.746 | 1 | 0.564 |
   | 1000 | 0.90 | 2.983 | 3.241 | 53 | 2.272 |
   | 1000 | 0.99 | 2.160 | 3.241 | 5 | 1.163 |

   No $K_{\rm eff}$ formula reproduces the observed values; the price is set by the extremal
   behaviour of the minimum of a dependent sequence.

   *Convergence.* In a converging solver the estimate freezes after a few iterations (§7.2), so the
   operative $n$ is the length of the transient. A law for $b_{n_{\rm eff}}$ in terms of the
   solver's convergence rate would connect §4 to practice and is the natural next theorem.

3. **The keep-mask as an unmodelled decision.** §7.5 shows that at a fixed threshold the retained
   set moves the reconstruction by up to $0.038$, while the bias/over-trust pair is blind to it.
   What is missing is a criterion on the keep-mask itself — a way to say, for a threshold schedule
   that is already front-on, which of the reachable retained sets is preferable — and an accounting
   of how that criterion composes with the frontier. This is not a coverage constraint; §7.5 finds
   the count effect small and the ordering dominated by which entries are kept.

4. **Threshold statistics with their own bias.** The theory prices taking a minimum; it does not
   price the bias of $T$. For MADN the two combine additively at first order, but not under
   contamination, where $T$ is the statistic §5 shows to be fragile.

5. **Multiple thresholds.** Section 4 treats a single scalar threshold, which is what all five
   methods analysed here carry. If a design were to anneal $d$ thresholds separately, the block of
   comparisons would be $dK$, but the price $b_{dK}$ would only apply under the additional
   requirement that the $d$ streams be independent and identically distributed — an assumption that
   tensor modes violate, both through shared factors and through differing noise scales. We
   therefore state no multi-threshold law, and record that one is needed.

6. **Optimal schedules under an explicit risk functional.** §6 chooses $\delta$ under three informal
   objectives. A minimax-regret formulation over an $\varepsilon$-contamination ball would give the
   choice a single justification, and is a finite-dimensional optimisation once (3.2) is available.

---

## Contributions

**(C1) Characterisation.** The min-annealing rule used across the robust-recovery literature is the
pointwise greatest element of the certificate cone (Prop. 2.2); it is *not* any isotonic projection
(Cor. 2.3), and its $\ell_\infty$ deviation is exactly twice that of the isotonic fit (Prop. 2.4).
It is the unique bias minimiser among schedules that never breach the licence constraint
(Cor. 2.6).

**(C2) Pricing.** Any admissible schedule satisfies $\textsf{bias}+\textsf{overtrust}\ge b_K$
(Thm. 3.1), the shifted running minimum attains the bound pathwise with an exchange rate of exactly
one (Thm. 3.2, Cor. 3.3), and the isotonic fits do not minimise this functional (Cor. 3.4). The
price has a closed quadrature (Thm. 4.1), an exact small-$K$ value (Prop. 4.2), an algebraic
heavy-tail law (Prop. 4.3), and an exact finite-sample form for a moving target (Thm. 4.5).

**(C3) Robustness and scope.** Under element-wise contamination there is no uniform positive
breakdown level in the block length (Prop. 5.1), and the frontier admits feasible hardening with a
computable cost (Prop. 5.2). On a real solver the rule is reproduced exactly, the price is governed
by the effective history rather than the iteration count (§7.2), and a controlled experiment
establishes that the threshold path does not determine the retained set (§7.5). This last item
delimits the theory; it is not a claim of improved recovery.

---

## Appendix: reproduction

| file | produces |
|---|---|
| `verify_core.py` | §2 (maximality, counterexample, factor 2), §3 (frontier, exchange, isotonic comparison), §4.1–4.3, §5 |
| `exp/code/e9b_terminal_class.py` | §2.1 (the terminal licence is implied; MSE of the unconstrained estimators) |
| `verify_final.py` | §4.1 (exact integral vs simulation), §3 pointwise checks, §4.2 |
| `verify_closing.py` | §4.4 lower bound (4.3), §8 item 2 (AR(1)), §5 RMSE table, causality |
| `verify_linf_identity.py` | Proposition 2.4 to machine precision |
| `verify_regen.py` | §4.3 and §4.4 regenerated simulation columns, §4.2 expansion-error table |
| `exp/code/e12_breakdown_corrected.py` | §5 Prop. 5.2 table (correct floor $b_{200}=2.746$) |
| `exp/code/e10_robust_estimation_link.py` | §5.1 influence functions and the rejection-point path |
| `exp/code/pbcd.py` | the solver of §7 (proximal block-coordinate step, MADN rule) |
| `exp/code/e3_solver_v2.py` | §7.4 relaxation scan |
| `exp/code/e5c_controlled.py` | §7.5 controlled keep-mask comparison |

Simulation columns are cross-checks against quadrature, never the other way round: when the two
disagree beyond $2$ s.e., the simulation is presumed at fault until the quadrature is shown
otherwise. Seeds: $20261102$ for all §7 tables, $112358$ for the exchange table, $60601$ for the
$L_\infty$ identity. `verify_draft_numbers.py` is the gate and must pass before submission.
