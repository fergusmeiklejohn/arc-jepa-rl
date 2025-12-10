LeJEPA: Provable and Scalable
Self-Supervised Learning Without the Heuristics

Randall Balestriero1,2,* Yann LeCun3,2,*

1 Brown University

3 New York University (NYU)

2 Meta-FAIR

* Equal contribution

Learning manipulable representations of the world and its dynamics is central to AI. Joint-Embedding
Predictive Architectures (JEPAs) offer a promising blueprint, but lack of practical guidance and theory has led to
ad-hoc R&D. We present a comprehensive theory of JEPAs and instantiate it in LeJEPA, a lean, scalable, and
theoretically grounded training objective. First, we identify the isotropic Gaussian as the optimal distribution
that JEPAs’ embeddings should follow to minimize downstream prediction risk. Second, we introduce a novel
objective–Sketched Isotropic Gaussian Regularization (SIGReg)–to constrain embeddings to reach that ideal
distribution. Combining the JEPA predictive loss with SIGReg yields LeJEPA with numerous theoretical and
practical benefits: (i) single trade-off hyperparameter, (ii) linear time and memory complexity, (iii) stability across
hyper-parameters, architectures (ResNets, ViTs, ConvNets) and domains, (iv) heuristics-free, e.g., no stop-gradient,
no teacher–student, no hyper-parameter schedulers, and (v) distributed training-friendly implementation requiring
only ≈50 lines of code. Our empirical validation covers 10+ datasets, 60+ architectures, all with varying scales and
domains. As an example, using imagenet-1k for pretraining and linear evaluation with frozen backbone, LeJEPA
reaches 79% with a ViT-H/14. We hope that the simplicity and theory-friendly ecosystem offered by LeJEPA will
reestablish self-supervised pre-training as a core pillar of AI research (GitHub repo).

Full FT

Frozen

Method

1-sh Full

1-sh Full

LeJEPA (in-domain)

ConvNeXt-V2 Nano 29.42 82.72 28.74 76.52
24.27 83.28 31.08 78.17
ResNet-34

Frontier (transfer)

DINOv2 ViT-S/16
DINOv3 ViT-S/16

21.05 78.34 27.68 67.62
24.71 81.60 30.17 71.38

Figure 1. LeJEPA overview. Top-left: Training loss exhibits strong correlation with downstream linear probe performance on ImageNet-1k
(ViT-base), providing the first practical loss for model selection without supervised probing. Top-right: Training stability without heuristics
even on 1.8B ViT-g models, stable training loss. Bottom-left: PCA features from ImageNet-1k pretrained LeJEPA ViT-Large demonstrate clear
semantic relationships. Bottom-right: Galaxy10 in-domain results showcasing LeJEPA’s in-domain pretraining consistently outperforms
state-of-the-art frontier foundation models transfer learning (DINOv2/v3 trained on natural images) across data regimes from 1-shot to full
supervision. This demonstrates that domain-specific SSL beats generic transfer learning, even against massive-scale frontier models, when the
framework scales effortlessly to any domain, model, and data scale.

5
2
0
2

v
o
N
4
1

]

G
L
.
s
c
[

3
v
4
4
5
8
0
.
1
1
5
2
:
v
i
X
r
a

0204060Testacc.(%)100101Trainloss(log-scale)Spearmancorr.:94.52%(ViT/base-8inet1k)λ0.040.080.120.160.2001428435772Epoch02468Loss0204060Accuracy(%)ViT-g/14,ImageNet-1K,LeJEPA

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

1 Introduction
Learning manipulable representations of the world and
its dynamics is a long-standing question in AI, with roots
dating back centuries ago [Von Helmholtz, 1867, Tolman,
1948, Gregory, 1980, Sutton, 1991, Friston, 2010]. Across
domains, e.g., image recognition, robotics, physics, space
exploration, the unifying question is how to learn an orga-
nized and actionable high-dimensional embedding space from
observations? Using Deep Networks–parameterized non-
linear operators 𝑓𝜽–to map observations to embeddings
is a standard first piece of that puzzle [LeCun et al., 2015,
Goodfellow et al., 2016]. The second, less standardized,
piece of that puzzle is how to train 𝑓𝜽. Joint-Embedding
Predictive Architectures (JEPAs) suggest training 𝑓𝜽 by
maximizing predictive agreement between the embed-
dings of semantically related views [Bromley et al., 1993,
LeCun, 2022, Balestriero et al., 2023]. Views can come
in two forms: transformations or corruptions. They can
involve masking, cropping, blurring, temporal or spatial
translations, geometric or photometric transformations,
viewpoint changes, views from different sensor modali-
ties, etc. The supervised forms involve human-produced
components such as image-caption pairs, text-code pairs,
etc [Tian et al., 2020].
In any case, views are expected
to share some degree of semantic relationship to allow
the prediction task to align 𝑓𝜽’s embeddings towards the
underlying knowledge present in the data.

Alas, JEPA’s prediction task admits failure modes, such
as representation collapse, where 𝑓𝜽 maps all inputs to
nearly identical embeddings (complete collapse) or to a low-
dimensional subspace (dimensional collapse) [Jing et al.,
2021][Jing et al., 2021, Cosentino et al., 2022, Balestriero
and LeCun, 2022]. To mitigate such shortcut solutions,
state-of-the-art recipes rely on heuristics–stop-gradient
[Chen et al., 2020a], asymmetric view generation [Wang
et al., 2022], teacher–student networks with carefully tuned
EMA schedules [Caron et al., 2021, Tian et al., 2021], ex-
plicit normalization and whitening layers [Ermolov et al.,
2021, Chen et al., 2021]–and a delicate balance of hyperpa-
rameters. As a result, today’s JEPA training is brittle and
most research has shifted toward scaling data [Vo et al.,
2024], models [Fan et al., 2025] and even post-training Ro-
das et al. [2025] while leaving the theoretical foundations
of JEPAs largely unexplored.

Our study proposes to break that cycle by question-
ing some of the fundamental design principles under-
pinning JEPAs. That introspection will start by asking

what are the necessary conditions that JEPAs should abide
by? Those minimal conditions will then act as axioms
for us to design a novel and lean JEPA. We identify two
axioms: (i) solving the prediction task while (ii) enforc-
ing an isotropic Gaussian distribution of the embeddings

(Section 3). While (i) follows standard practice [Balestriero
and LeCun, 2022], we introduce in Section 4 a novel dis-
tribution matching objective–Sketched Isotropic Gaussian
Regularization (SIGReg)–to enforce (ii). The use of SIGReg
not only removes the need for the numerous heuristics
previously employed to prevent representation collapse,
but SIGReg also exhibits favorable scaling properties as its

memory and computational complexity is linear in dimension
and sample size. Crucially, SIGReg’s isotropic Gaussian
enforcement solves the collapsed shortcut solution and
provably minimizes the model’s expected risk over the
space of downstream tasks to be encountered post-training.
The resulting JEPA solution–coined Latent-Euclidean JEPA
(LeJEPA)–is introduced in Section 5. Beyond theoretical
optimality, LeJEPA offers numerous benefits such as (i)
provable statistical guarantees, (ii) removal of heuristics
such as teacher-student networks, (iii) linear memory and
computational complexity, and most importantly (iv) a
unified design with a single trade-off parameter that works
out of the box across datasets, architectures and scales (see
Section 6). We summarize our contributions below.

Contribution 1: We prove the optimal embedding
distribution for foundation models. We establish that
the isotropic Gaussian uniquely minimizes downstream
prediction risk across broad task families. In Section 3, we
derive this result rigorously for both linear (Section 3.1)
and nonlinear probes (Section 3.2), providing the first
principled answer to what distribution 𝑓𝜽’s embeddings
should follow. This theoretical result transforms JEPA
design from heuristic exploration to targeted optimization.
Contribution 2: We introduce SIGReg, a distribution
matching objective that uniquely combines provable
correctness with computational efficiency at scale. We
present Sketched Isotropic Gaussian Regularization (SIGReg),
a novel objective that enforces distributional alignment
via random projections and characteristic-function match-
ing (Section 4 and Figure 2). SIGReg provides statistical
guarantees (Sections 4.1 and 4.2) while achieving linear
complexity and bounded gradients—a combination that
existing distribution matching methods do not offer. Criti-
cally, its projection-based construction defeats the curse
of dimensionality (Section 4.3), making it both theoreti-
cally sound and practically efficient for high-dimensional
embeddings.

Contribution 3: We design LeJEPA, a statistically op-
timal JEPA that eliminates collapse by construction. By
combining JEPA’s predictive objective with SIGReg target-
ing the isotropic Gaussian, we introduce LeJEPA—Latent-
Euclidean JEPA (Section 5). LeJEPA requires only a single
hyperparameter, eliminates representational collapse with-
out stop-gradients or teacher-student architectures, and
transfers across architectures and datasets without hy-
perparameter tuning. This demonstrates that principled

2

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

𝑓𝜽
→

Figure 2. Sketched Isotropic Gaussian Regularization (SIGReg): Given some arbitrary input data with density 𝑝𝑥 with support that may or may
not lie on a manifold (left), a Deep network (DN) encoder ( 𝑓𝜽) produces embeddings 𝒛 = 𝑓𝜽(𝒙) with some distribution 𝒛 ∼ 𝑝𝑧 (middle). Our proposed
Backward Cramér-Wold Statistics (Section 4) objective pushes 𝑝𝑧 to match a target distribution 𝑝𝑡 by projecting the embeddings along 1𝑑 directions
(middle, arrows) and enforcing that the univariate densities (right, colored lines) match the distribution of 𝑝𝑡 , projected along the same directions.
Any popular statistical test (provided in Section 4.2) can assess the goodness-of-fit–in practice we argue for characteristic function tests (Section 4.2).
By using SIGReg with 𝑝𝑡 isotropic Gaussian (right, black lines), we introduce a lean and provably optimal (Section 3) JEPA, coined LeJEPA, free of
numerous heuristics and able to produce competitive performances (Sections 5 and 6).

theory directly yields practical simplicity.

Contribution 4: We validate LeJEPA at scale across
diverse architectures and establish in-domain pretrain-
ing as viable. Our experiments (Section 6) span ViTs,
ConvNeXts, ResNets, MaxViTs, and Swin Transformers
at scales approaching 1 billion parameters, where LeJEPA
matches or exceeds state-of-the-art methods while main-
taining training simplicity and robustness. Critically, on
domain-specific datasets (Galaxy10, Food101), LeJEPA
outperforms DINOv2-based transfer learning when pre-
trained directly on target data. This challenges the transfer
learning paradigm and demonstrates that principled SSL
can unlock effective in-domain pretraining—previously
considered impractical for small datasets.

2 Background and Notations
We start by introducing some of the notations we will be
using throughout our manuscript (Section 2.1), followed
by a review of JEPAs (Section 2.2), and existing literature
studying their design (Section 2.3).

2.1 Notations and Definitions
Data. We are in possession of a dataset of shape (𝑁 , 𝑉 , 𝐷) ∈
N∗3 where 𝑁 is the number of samples, 𝑉 is the number
of views, and 𝐷 is the dimension. One entry of this
dataset is accessed via 𝒙𝑛,𝑣,𝑑. Those dimensions are often
interpreted as follows: (N) is the number of independent
samples, e.g., different images or different videos, (V) is
the number of views, e.g., data-augmentations for images,
frames for videos, and (D) is the dimension of each 𝒙𝑛,𝑣,
e.g., number of RGB pixels for images.
In many cases
the ordering over 𝑉 is given by time–but in some cases,
e.g., data-augmentation of an image, ordering becomes

irrelevant. Our study does not require any particular
choice to organize one’s dataset into a (𝑁 , 𝑉 , 𝐷) tensor–

and none of our theory and implementation assumes a particular
design decision for that tensor. However, we will rely on
the following two properties, (independence) the samples
𝒙𝑛 , 𝒙𝑛′ have been obtained independently from each other
∀𝑛 ≠ 𝑛′, and (identically distributed) the sampling process
was identical among 𝒙𝑛 , ∀𝑛.

Deep Networks. Today’s AI solutions rely on Deep
(Neural) Networks (DNs), which are compositions of a large
number of parameterized linear and nonlinear operators.
We denote the DN’s mapping as 𝑓𝜽 : R𝐷 → R𝐾 with 𝐾
the dimension of the embedding space. The internals
of 𝑓𝜽 are designed by the researcher to incorporate as
much prior knowledge about the data as possible. The
details of 𝑓𝜽 are irrelevant to our study–as we will see
the proposed LeJEPA works out-of-the-box on any 𝑓𝜽.
In any case, all the learnable parameters are gathered in
the vector 𝜽 ∈ R𝑃, with 𝑃 counting the total number of
parameters. A central challenge in AI research is to design
the right architecture and training objective so that 𝜽 can
be learned from gradient descent to ultimately produce a
useful system, or foundation model, 𝑓𝜽.

JEPAs. A foundation model is any system, e.g., a DN,
able to solve numerous downstream tasks without requir-
ing any change in its internal parameters 𝜽. This is in sharp
contrast with a supervised model that only considers its
training task. JEPAs have formally been introduced by
LeCun [2022] as a vehicle to produce foundation models.
The core building blocks of JEPAs rely on numerous well-
established techniques such as siamese networks [Bromley
et al., 1993] and predictive coding [Helmholtz et al., 1867,
Bruner and Postman, 1949]. While the exact blueprint of

3

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Definition 1: JEPA

JEPA(𝒙) ⇐⇒ Enc (cid:0)𝒙𝑛,𝑡+1,.(cid:1) is predictable from Enc (cid:0)𝒙𝑛,𝑡,.(cid:1) , ∀𝑛, 𝑡 and Enc (cid:0)𝒙.,.,.(cid:1) is not degenerate.

(1)

JEPAs varies greatly between use-cases, they all rely on
two core principles: (i) being able to predict the embed-
ding of a view 𝒙𝑛,𝑣 from the embedding of another view
𝒙𝑛,𝑣′ , 𝑣′ ≠ 𝑣, all while (ii) ensuring that the embeddings
do not become degenerate. Concretely, once a JEPA is
designed and trained, it should be able to solve numer-
ous downstream tasks in zero or few shots. The JEPA
objective function, along with some examples for 𝒙, is
provided in Equation (1). The predictability criterion can be
done by directly comparing the embeddings of the partial
views 𝐸𝑛𝑐(𝒙𝑛,𝑣,.) and 𝐸𝑛𝑐(𝒙𝑛,𝑣′,.) with a metric, e.g., ℓ𝑝. In
some cases, an additional DN coined Pred, is employed
to compare 𝑃𝑟𝑒𝑑(𝐸𝑛𝑐(𝒙𝑛,𝑣,.)) against 𝐸𝑛𝑐(𝒙𝑛,𝑣′,.)–which is
only justified when there exists an asymmetry between
the information content of the different views, e.g., by
conditioning the predictions on observed actions from
robotics data [Khazatsky et al., 2024].

2.2 The Need for Reliable Pretraining
The JEPA’s prediction task is designed based on a priori
knowledge of the data. Its design is often quite natural
since it is relatively intuitive to form 𝒙 so that its views
share the relevant information content one hope to capture.
On the other hand, the design of the “anti-collapse” crite-
rion is much closer to a game of Whac-A-Mole. Today’s
designs rely on many different under-specified safeguards
which are carefully combined in the hope that degener-
ate shortcut solutions are avoided during training. Such
mechanisms include (i) feature whitening [Ermolov et al.,
2021, Bardes et al., 2021], (ii) negative samples [Chen
et al., 2020a, He et al., 2020], and (iii) asymmetric views
and teacher-student networks with stop-gradient [Caron
et al., 2021, Assran et al., 2023]. Those mechanisms all
suffer from at least two of the following limitations: (i)

under-specification, i.e., the criteria can be minimized
while embeddings are in a degenerate configuration, (ii)
quadratic time and memory complexity with mini-batch
size and/or embedding dimension, (iii) sensitivity to data
distribution, hyperparameters, architecture, and (iv) lack
of theoretical understanding and guarantees.

2.3 The Need for Actionable Theory
For decades, the two major solutions for AI were super-
vised learning [LeCun et al., 2015] and learning by recon-
struction [Rumelhart et al., 1986]–sometimes combined
together, e.g., for semi-supervised learning [Kingma et al.,
2014]. In supervised learning, the labels both ensure that
semantically similar samples are close to each other in em-
bedding space while preventing complete representation
collapse. In particular, it is possible to measure the amount
of collapse in supervised learning as a function of the num-
ber of classes [Papyan et al., 2020]. The reconstruction
objective is similarly well suited to prevent representation
collapse as the original input must be recovered from the
embeddings, i.e., the embeddings must be as informative
about the input as possible–up to some optional denoising
tasks that users can setup as part of the training [Vincent
et al., 2010].

Because supervised and reconstruction-based learning
have been widely studied for decades, there exists a large
body of work to explain and inform practical designs–as
well as studying their limitations in producing foundation
models [Balestriero and LeCun, 2024, Van Assel et al.,
2025]. This is not the case for the more recent JEPAs
where empirical advances quickly outpace anyone hoping
to delve into their inner workings. This dynamic led the
community to focus on post-hoc theoretical justification
of already found solutions [Liu et al., 2021, Shwartz Ziv

4

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

and LeCun, 2024, Shwartz-Ziv et al., 2022, Zhang et al.,
In most cases, those studies involve the Mutual
2023].
Information (MI) [Shannon, 1948, Cover, 1999] whose dif-
ferent bounds recover established methods [Gutmann and
Hyvärinen, 2010, Ma and Collins, 2018, Oord et al., 2018,
Poole et al., 2019, Hjelm et al., 2018, McAllester and Stratos,
2020]. Because existing studies focus on explaining and
interpreting already developed JEPAs, too little principled
guidance and innovation has been brought forward. In-
stead, most of the recent empirical advances take the form
of collecting larger dataset, scaling up pre-existing training
recipes [Goyal et al., 2019, Chen et al., 2020b, Oquab et al.,
2023, Fan et al., 2025], and deriving novel data curation
processes [Vo et al., 2024, Kerdreux et al., 2025].

In contrast, our goal in the following Sections 3 to 5 will
be to derive a novel JEPA solution from first principles, i.e.,
whose design relies on proved necessary conditions for
optimality, and with a pretraining recipe that can finally
reconcile exploratory research, scalability, and state-of-the-
art performances.

3 Latent Euclidean: Embeddings
Should be Isotropic Gaussian
We address a fundamental question: which distribution
should Enc(𝒙) follow to minimize empirical risk on any down-
stream task? We prove that the isotropic Gaussian is the
unique optimal distribution for both linear (Section 3.1)
and nonlinear probing (Section 3.2), with geometric in-
tuition provided in Section 3.3. This theoretical result
establishes the necessary design principle for our JEPA;
Section 4 then provides the practical implementation to
achieve it.

3.1 Linear Probing
We begin by identifying the optimal distribution for 𝑓𝜽’s
embeddings by analyzing linear probes–one of the most
popular methods for frozen encoder evaluation. Specif-
ically, we ask: which distribution for 𝑓𝜽(𝒙) would be most
favorable for solving arbitrary downstream tasks, i.e., for any
realization of targets 𝒚?

Denote as 𝒁 ∈ R𝑁×𝐾 the matrix of 𝑁 embeddings, each
𝐾-dimensional, from 𝑓𝜽(𝒙𝑛). The unknown corresponding
labels are denoted as 𝒚 ∈ R𝑁 . Without loss of generality, we
consider univariate targets; the following analysis extends
to multivariate targets. The linear probe minimizes the
following least square problem [Bishop and Nasrabadi,
2006]

ˆ𝛽 = arg min

𝛽∈R𝐾

∥𝒚 − 𝒁𝛽∥2

2 + 𝜆∥𝛽∥2
2,

(OLS)

where ˆ𝛽 is the optimal probe parameters, and 𝜆 ≥ 0
is an hyperparameter controlling the Tikhonov regular-
izer strength [Bishop, 1995, Golub et al., 1999]. Despite

not knowing 𝒚, it is possible to describe the bias and
variance of the estimator ˆ𝛽 as a function of the distri-
bution of 𝒁. Consider two embeddings with identical
column spans 𝒁aniso, 𝒁iso. 𝒁aniso’s covariance matrix eigen-
values are given by {𝜆𝑘}𝐾
with at least two distinct
𝑘=1
values, while 𝒁iso’s covariance matrix eigenvalues are all
equal to 1
𝑘=1 𝜆𝑘. Hence, the two candidate embeddings
𝐾
𝒁aniso, 𝒁iso capture the same intrinsic features and have
same energy, but different geometries.

(cid:205)𝐾

Lemma 1: Anisotropy amplifies bias

Whenever 𝜆𝐾 > 𝜆1, there always exists a downstream task
(𝒚) for which 𝒁aniso produces a higher bias estimator than
𝒁iso for 𝜆 > 0. (Proof in Section B.1.)

Lemma 2: Anisotropy amplifies variance

With 𝜆 = 0, the total variance of ˆ𝛽 (OLS) is minimized for 𝒁iso
with tr(Var( ˆ𝜷aniso)) > tr(Var( ˆ𝜷iso)). (Proof in Section B.2.)

From the above lemmas. 1 and 2 we obtain that the
distribution of features must be isotropic. We now move
to nonlinear probing where the standard Gaussian will
emerge as the unique optimum.

3.2 Nonlinear Probing
To allow for more flexible evaluation of the pretrained
encoder 𝑓𝜽, it has become increasingly common to work
with a nonlinear probe. We analyze two widely-used
nonlinear methods: radius-based k-NN [Taunk et al., 2019,
Sun and Huang, 2010, Zhang et al., 2017, Abu Alfeilat et al.,
2019] for its simplicity and kernel methods [Nadaraya,
1964, Watson, 1964] for their theoretical tractability.

As in Section 3.1, we ask ourselves which distribution of
embeddings would be preferable for a foundation model.
We first define our prediction function. The training data
consists of the 𝑁 embeddings along with their training
labels {(𝒛𝑛 , 𝒚𝑛)}𝑁
. The prediction, using radius-based
k-NN for a query vector 𝒒 is formed as

𝑛=1

(cid:98)𝒚(𝒒) :=

1
|𝒩𝑟0(𝒒)|

(cid:213)

𝒚𝑛 ,

𝑛∈𝒩𝑟0

(𝒒)

(kNN)

where 𝒩𝑟0(𝒒) = {𝑛 : ∥𝒛𝑛 − 𝒒∥ ≤ 𝑟0}. The specific choice
of radius 𝑟0 controls how many neighbors predictions are
averaged to form the query’s prediction. The kernel’s
prediction at a query 𝒒 ∈ R𝐾 is given by

(cid:98)𝒚(𝒒) ≜

(cid:205)𝑁

𝑛=1 𝐾 ℎ(𝒒 − 𝒛𝑛)𝒚𝑛
(cid:205)𝑁
𝑛=1 𝐾 ℎ(𝒒 − 𝒛𝑛)

.

(Kernel)

We search over all distributions of Z subject to a fixed to-
tal variance constraint, e.g., Tr(Cov(𝒁)) = 𝜅1 or ∥Cov(𝒁)∥𝐹 =
𝜅2. The specific value of 𝜅 does not affect the optimal dis-

5

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

parameters equals 1 only for isotropic distributions, de-
grading for anisotropic cases regardless of sample size or
regularization strength. Regarding variance (lemma. 2),
we show in Figure 3 that learned parameters vary sig-
nificantly more across training sets when the covariance
is anisotropic (right) compared to isotropic (left)—even
when using logistic regression instead of OLS. Figure 17
further illustrates this effect, showing the distribution of
learned 𝛽 parameters across different training samples for
both cases. The anisotropic distribution clearly produces
higher-variance estimators.

These theoretical and empirical results establish our
design principle for LeJEPA: embeddings 𝑓𝜽(𝒙) should follow
an isotropic Gaussian distribution to minimize worst-case risk
across downstream tasks encountered post-training. Section 4
introduces a novel regularizer to achieve this distribution.

4 SIGReg: Reliable Isotropic
Gaussian Regularization in
High-Dimension

Having established the isotropic Gaussian as the optimal
embedding distribution (Section 3), we now introduce
Sketched Isotropic Gaussian Regularization (SIGReg)–a dis-
tribution matching objective that is simultaneously (i)
differentiable, (ii) scalable, (iii) provable, and (iv) interpretable.
SIGReg builds on three key innovations. First, we formu-
late distribution matching as a statistical test under the null
hypothesis 𝑃𝜽 = 𝑄 (Section 4.1). Second, we identify a test
that guarantees bounded gradients and curvature while
maintaining linear complexity and efficient multi-GPU
scaling (Section 4.2). Third, SIGReg bypasses the curse of
dimensionality, eliminating collapsed shortcut solutions
entirely (Section 4.3).

4.1 Hypothesis Testing as a Judge
Asking for 𝑓𝜽(𝒙)’s distribution 𝑃𝜽 to match a target distri-
bution 𝑄 is typically done by creating various measures
of distance or divergence, and estimating them in high-
dimension. We propose a different starting point grounded
in statistics. Consider the hypothesis testing framework
[Fisher, 1928, Neyman and Pearson, 1933] given by

𝐻0 : 𝑃𝜽 = 𝑄 vs. 𝐻1 : 𝑃𝜽 ≠ 𝑄,

(2)

with 𝐻0 being referred to as the null hypothesis. That is,
we are asking in Equation (2) if there is enough empiri-
cal evidence to reject the null. To answer that question,
one (i) employs a test-statistic, i.e., a single scalar value
summarizing the evidence from the empirical samples, (ii)
determines a critical value 𝜏𝛼 for the test-statistic based on
the probability 𝛼 of Type I error, i.e., of mistakenly rejecting
a true null hypothesis, (iii) compares the test-statistic to

Figure 3. Illustration of lemma. 2 showcasing how anisotropic (right)
embeddings lead to higher variance estimator compared to isotropic
embeddings (left). We sample 100 training points for the 2-class clas-
sification task and fit a logistic regression–repeating the process over
numerous training set sample. Each sampling results in a decision
boundary (purple).

tribution shape. Following the same type of derivations
as done in the linear regime–with the exception of some
additional regularity conditions–we are able to precisely
identify the isotropic Gaussian as the unique optimum to
minimize bias as formalized below.

Theorem 1: isotropic Gaussian Optimality

The integrated square bias (ISB) over query points is given by

ISB𝑘-NN =

𝑟4
0
(𝐾 + 2)2

𝑔 𝐽(𝑝) + 𝑂(𝑟4
𝜏2
0 ),

(k-NN)

ISBkernel ≤

(cid:16) ℎ2𝜇2(𝐾)
2

(cid:17) 2 (cid:16)

2𝐵2 + 8𝐿2𝐽(𝑝)

(cid:17)

+ 𝑜(ℎ4),

(kernel)

and among distributions with a scalar-based covariance con-
straint, the isotropic Gaussian is the unique minimizer of the
integrated square bias. (Proof in Sections B.4 and B.7.)

Numerous additional details and discussions on the
regularity assumptions we employed are provided in Sec-
tion A. Together, these results establish the isotropic Gaus-
sian distribution as the optimal design to minimize the
worst-case risk of a foundation model across downstream
tasks.

3.3 Geometric and Practical Insights
We now empirically validate that the isotropic Gaussian is
optimal when no information about downstream tasks is
available. We focus on linear probing (Section 3.1), where
all considered distributions have the same total variance.
When employing a linear probe, an anisotropic distri-
bution increases both bias (with Tikhonov regularization)
and variance. Examining bias first (lemma. 1), we present
in Figure 18 visualizations for both continuous regres-
sion and discrete classification tasks. We observe that
the cosine similarity between estimated and ground-truth

6

−4−2024x1−4−2024x2Isotropic,Var(ˆβ)=0.0056TrueboundaryLearnedboundaries−4−2024x1Condition#:20Anisotropic,Var(ˆβ)=0.0801TrueboundaryLearnedboundariesLeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

the critical value 𝜏𝛼; if the test-statistic exceeds 𝜏𝛼, reject
the null hypothesis. If the null is not rejected, we can only
claim that there is not sufficient empirical evidence against
𝑃𝜽 = 𝑄.

As it stands, Equation (2) remains impractical in large
dimension as existing tests have at least quadratic complex-
ity with the number of samples considered (more details in
Section F). We thus propose to derive a sketching strategy
by decomposing Equation (2) into simpler univariate tests.
Denoting the push-forward distributions 𝑃(𝒂)
≜ (𝒂⊤)#𝑃𝜽
𝜽
and 𝑄(𝒂) ≜ (𝒂⊤)#𝑄, we can define the following directional
univariate test

𝐻0(𝒂) : 𝑃(𝒂)

𝜽 = 𝑄(𝒂) vs. 𝐻1(𝒂) : 𝑃(𝒂)

𝜽 ≠ 𝑄(𝒂),

(3)

for a given directional unit-norm vector 𝒂 ∈ 𝒮 𝐾−1. The
corresponding directional test-statistic of Equation (3) is
computed as 𝑇({𝒂⊤ 𝑓𝜽(𝒙𝑛)}𝑁
𝑛=1). Examples of tests 𝑇 will
be provided in the later Section 4.2. Repeating that process
over a set of 𝑀 directions A ≜ {𝒂1, . . . , 𝒂𝑀} and aggre-
gating the individual values lead to the following global
test-statistic

𝑇A({ 𝑓𝜽(𝒙𝑛)}𝑁

𝑛=1) ≜ max
𝒂∈A

𝑇({𝒂

⊤ 𝑓𝜽(𝒙𝑛)}𝑁

𝑛=1).

(4)

We now provide a formal statement asserting the consis-
tency of Equation (4) to test the original multivariate null
hypothesis from Equation (2). Our result leverages the
well-known union-intersection principle [Roy, 1953], and
a slightly modified Cramér-Wold theorem. We denote by
𝑑
= equality in distribution.

Lemma 3: Hyperspherical Cramér-Wold

Let 𝑋 , 𝑌 be R𝑑-valued random vectors, then

⟨𝒖, 𝑋⟩

𝑑
= ⟨𝒖, 𝑌⟩, ∀𝒖 ∈ S𝑑−1 ⇐⇒ 𝑋

𝑑
= 𝑌.

Convergence in distribution also holds. (Proof in Section B.8.)

Theorem 2: Sufficiency of directional tests

Equation (4) is a valid statistical test for Equation (3) as

𝑃 = 𝑄 =⇒ lim sup
𝑛→∞

Pr

𝑃 ≠ 𝑄 =⇒ lim sup
𝑛→∞

Pr

(cid:16)

(cid:16)

𝑇A({ 𝑓𝜽(𝒙𝑛)}𝑁

𝑛=1) ≥ 𝜏𝛼

𝑇A({ 𝑓𝜽(𝒙𝑛)}𝑁

𝑛=1) ≥ 𝜏𝛼

(cid:17)

(cid:17)

≤ 𝛼, (level)

= 1, (power)

(Proof in Section B.9.)

The assumptions required in the proof of thm. 2 hold
for classical consistent univariate tests 𝑇 such as the ones
presented in the following Section 4.2.

7

Figure 4. Examples of distributions living on the surface of the sphere
with varying Sobolev smoothness coefficients 𝛼. As per thm. 5, the
greater 𝛼 is, the more global will be the impact of SIGReg for a given
number of directions 𝑀. Practically, this represents the distribution of
the encoder’s output. Because the target density (isotropic Gaussian) is
smooth, the 𝛼 coeffcients of the embedding will quickly grow hereby
making SIGReg (def. 2) immune to the curse of dimensionality.

4.2 SIGReg: Sketching the Epps-Pulley

Test is Stable and Scalable

Our proposed regularizer–coined Sketched Isotropic Gaus-
sian Regularization (SIGReg)–follows directly from thm. 2
using any statistical test 𝑇 targeted towards the isotropic
Gaussian, illustrated in Figures 2 and 5, and formalized
below.

Definition 2: SIGReg (PyTorch code in algorithm 1)

SIGReg sketches a statistical test 𝑇 towards isotropic Gaussian

SIGReg𝑇 (A, { 𝑓𝜽(𝒙𝑛)}𝑁

𝑛=1) ≜

1
|A|

(cid:213)

𝒂∈A

𝑇({𝒂

⊤ 𝑓𝜽(𝒙𝑛)}𝑁

𝑛=1),

(SIGReg)

where we recommend the Epps-Pulley test (Section 4.2.3) for
𝑇.

We replace the maximum over 𝒂 ∈ A in thm. 2 by
an average in (SIGReg) to avoid sparse gradient over the
directions in A. We now delve on the choice of 𝑇 for
which we compare well-known candidate tests in the field
of statistics that are categorized into (i) moment based
(Section 4.2.1), (ii) CDF based (Section 4.2.2), and (iii) CF
based (Section 4.2.3) statistics–ultimately justifying our
choice of the Epps-Pulley statistic.

4.2.1 Moments are Unstable and Insufficient

The first family of statistics we consider are moment-based.
Taking the standard Gaussian as an instanciation for the
moments, we can define the Jarque-Bera [Jarque and Bera,
1980] test that compares the third and fourth moments,

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Figure 5. Constructed data density with “X” distribution whose marginals are standard Gaussian and whose covariance is identity
(left densities). Applying 𝑀 = 10 projections on the half circle directions produces 10 univariate distributions that can be compared
against a standard Gaussian (left) using any preferred statistic from Section 4.2. The appropriate direction is able to capture the
degenerate distribution of the data hereby creating a spike in the statistic value.

i.e., skewness and kurtosis, as

Theorem 3: Insufficiency of K Moments

(cid:154)skew(𝒖)2 +

(cid:33) 2

(cid:32)

(cid:100)kurt(𝒖) − 3
2

JB(𝒖) ≜

𝑁
6

(cid:169)
(cid:173)
(cid:171)

,

(Jarque-Bera)

(cid:170)
(cid:174)
(cid:172)

Minimizing the following objective with 𝑐𝑘 > 0, ∀𝑘

𝑐𝑘

(cid:16)

𝑚𝑘

(cid:17)

(cid:16)

𝑃(𝒂)
𝜽

− 𝑚𝑘

(cid:16)

𝑄(𝒂)(cid:17)(cid:17) 2

,

𝐾
(cid:213)

𝑘=1

1
𝑛

1
𝑛

𝑖=1(𝑥𝑖 − ˆ𝜇)3
(cid:205)𝑛
ˆ𝜎3

and (cid:100)kurt is the kurtosis

where (cid:154)skew is the skewness computed from the data as
. Typically,
the (Jarque-Bera) test is used to see if a density follows a
Gaussian distribution of any mean and variance–hence it
only looks at moments 3 and 4. In our case we aim for a
standard Gaussian test and thus add the usual statistics
on the first two moments, leading to the extended test

𝑖=1(𝑥𝑖 − ˆ𝜇)4
(cid:205)𝑛
ˆ𝜎4

EJB(𝒖) ≜

𝑁 ˆ𝜇(𝒖)2
ˆ𝜎(𝒖)2

+

(𝑁 − 1) (cid:0) ˆ𝜎(𝒖)2 − 1(cid:1) 2
2
(Extended Jarque-Bera)

+ JB(𝒖).

The (Extended Jarque-Bera) acts as a moment matching
problem over the first four moments. Such moment match-
ing methods have proven powerful not only for statistical
tests but also as mean to learn parametric and nonpara-
metric models of data.

The Stability and Identifiability Conundrum. We now
explain why moment-based tests–albeit powerful–will
not be suited for LeJEPA. The 𝑘𝑡 ℎ of a distribution 𝑃
is denoted as 𝑚𝑘(𝑃). The first observation is that well-
behaved distributions abiding the Carleman’s condition
(cid:205)∞
𝑘=1 𝑚2𝑘(𝑄)−1/(2𝑘) = ∞ [Carleman, 1926], such as the Gaus-
sian, or for distributions with finite interval [Hausdorff,
1923] are uniquely determined by their moments. However,
using a finite number of moments creates the following
non-identifiability issue which well-known in statistics and
often used as a motivation to use all moments [Lehmann
and Romano, 2005].

for finite 𝐾 does not imply 𝑃(𝒂)

𝜽 = 𝑄(𝒂). (Proof in Section B.11.)

Hence thm. 3 prescribes us with the guideline to em-
ploy as large 𝐾 as possible to remove collapsed shortcut
solution by making sure our distribution matching is ac-
curate. Yet, doing so leads to unstable gradient-based
training due to the gradient norm scaling as 𝑂(𝑘), and
the variance of Monte Carlo gradient estimates growing
as 𝑂(𝑘2𝑚2(𝑘−1)) for the 𝑘-th moment since (cid:13)
)(cid:13)
(cid:13) =
∥E(cid:2)𝑘(𝒂⊤ 𝑓𝜽(𝒙))𝑘−1𝒂⊤𝐽 𝑓𝜽 (𝒙)(cid:3)∥, with 𝐽 𝑓𝜽 (𝒙) ∈ R𝐾×𝑃 the Jaco-
bian matrix–hereby creating an impractical situation where
training stability and identifiability can not be achieved
simultaneously.

(cid:13)∇𝜃𝑚𝑘(𝑃(𝒂)

𝜽

4.2.2 Cumulative Density Functions are Impractical

The second family of tests acts upon the CDF. Because those
tests require sorting, let’s denote the 𝑘th order-statistics of
𝑁 samples by 𝑥𝑘:𝑁 . Two highly standard tests are quadratic
Empirical Density Function statistics with different weight-
ing known as Cramér-von Mises [Cramér, 1928, Von Mises,
1981] and Anderson Darling [Anderson and Darling, 1952],
and given by

𝑇𝑤 = 𝑁

∫ ∞

−∞

(𝐹𝑁 (𝑥) − 𝐹(𝑥))2 𝑤(𝑥)𝑑𝐹(𝑥)

𝑤(𝑥) = 1,
𝑤(𝑥) = [𝐹(𝑥)(1 − 𝐹(𝑥))]−1,

(Cramér-von Mises)

(Anderson-Darling)

where 𝑤(𝑥) is a weighting function. Adding the 𝑈 2 statis-
tics on top of Equation (Cramér-von Mises) recovers the

8

−202x1(blue)—x2(red)050100150Count−202x1−202x2−50hx,aiii:0i:1i:2i:3i:4i:5i:6i:7i:8i:9p(hx,aii)i:0i:1i:2i:3i:4i:5i:6i:7i:8i:90.20.61.0‘1and‘2vcregextjarquebetawatsoncramervonmisesandersondarlingeppspulleyLeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Figure 6. 𝑁 = 100 samples are drawn from a 1024-dimensional standard Gaussian, and the first 2 coordinates are altered to produce
the “X” distribution from Figure 5 (left-most column). For each statistic (all other columns), we perform gradient descent on the
samples to minimize their value, at each iteration step with sample 𝑀 = 10 random directions to evaluate SIGReg (recall def. 2). We
obtain that albeit this is a high-dimensional distribution with limited number of samples, SIGReg is able to capture the degenerate
subspace and adapt the data accordingly to match an isotropic Gaussian distribution. Additional figures with varying dimensions
and number of 1d projections are provided in Figure 16.

Watson test [Watson, 1961]

𝑈 2 = 𝑇𝑤 − 𝑁

(cid:18)

¯𝐹 −

(cid:19) 2

.

1
2

(Watson)

We do not consider the Kolmogorov-Smirnov test [Kol-
mogorov, 1933] as it employs the ℓ∞-norm instead of the
ℓ2-norm hereby producing sparse gradients. Another
common test is the Shapiro-Wilk test [Shapiro and Wilk,
1965] which we found to be unstable in practice–details
are provided in Section E.

Lack of Scalability and Differentiability. CDF-based
tests require sorting that have been highly optimized, e.g.,
with the 𝒪(𝑁 log(𝑁)) Quicksort algorithm [Hoare, 1962]
but that nonetheless breaks the embarrassingly parallel na-
ture of SGD–especially on multi-GPU [Tanasic et al., 2013,
Maltenberger et al., 2022] due to synchronization require-
ments. Moreover, these tests involve non-differentiable
operations (sorting and order statistics), making them
unsuitable for gradient-based optimization without re-
laxations [Cuturi et al., 2019, Grover et al., 2019, Petersen
et al., 2022]. While there exists intricate sketching solutions
[Dunning and Ertl, 2019, Masson et al., 2019, Dunning,
2021], each of those solutions introduce numerous addi-
tional hyper-parameters–going against our first motivation
for LeJEPA.

4.2.3 Characteristic Functions are Stable, Scalable

and Identifiable

The third family of tests is concerned with Empirical Char-
acteristic Functions (ECF) which are the Fourier transform
of the density function. The Epps–Pulley test [Epps and
Pulley, 1983] is one of the most popular test and simply
compares in weighted ℓ2-norm the ECF of the data against
a target CF

𝐸𝑃 = 𝑁

∫ ∞

−∞

(cid:12)
(cid:12)

2
ˆ𝜙𝑋 (𝑡) − 𝜙(𝑡)(cid:12)
(cid:12)

𝑤(𝑡)𝑑𝑡.

(Epps–Pulley)

(cid:205)𝑛

The first crucial observation is that the ECF being defined
as ˆ𝜙𝑋 (𝑡) = 1
𝑗=1 𝑒 𝑖𝑡𝑋𝑗 is naturally differentiable and easily
𝑛
computed in distributed settings via efficient all_reduce
operations, as the ECF is a simple average of complex
exponentials. The weight function is typically Gaussian,
such as 𝑤(𝑡) = 𝑒−𝑡2/𝜎2 with 𝜎 commonly set to 1.

Other tests, e.g., based on the Entropy [Székely and
Rizzo, 2005] are not considered here as they require nu-
merous additional design choices for the univariate En-
tropy estimation [Silverman, 2018, Beirlant et al., 1997], e.g.,
using kernels [Joe, 1989], or M-estimators [Miller, 2003].

Epps-Pulley has bounded loss, gradient and curva-
ture. We now consider the remaining two families of tests:
moment-based and CF-based. First, recall that moments
are polynomial in the data and with extreme growth rate

9

05dim1−2024dim2originaldata05dim1VCReg05dim1ExtendedJarqueBera05dim1CramerVonMises05dim1Watson05dim1AndersonDarling05dim1EppsPulley−2.50.02.5dim3−4−2024dim4−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Algorithm 1. SIGReg with Epps-Pulley statistic with DDP support and
𝒪(𝑁) time and memory complexity. x is a (N, K) tensor, num_slices is
|A| in def. 2, ‘global_step‘ is used for sync. sampling across GPUs and
can be omited for single-GPU training. An optimized implementation
with caching is also provided in our official codebase, computation times
provided in Table 6.

def SIGReg ( x , g l o b a l _ s t e p , num_slices =256) :

g e n e r a t o r =g , ∗∗ dev )

# s l i c e sampling −− synced across d e v i c e s −−
dev = d i c t ( d e v i c e =x . d e v i c e )
g = t o r c h . Generator ( ∗ ∗ dev )
g . manual_seed ( g l o b a l _ s t e p )
proj_shape = ( x . s i z e ( 1 ) , num_slices )
A = t o r c h . randn ( proj_shape ,
A / = A . norm ( p=2 , dim =0)
# −− Epps− P u l l e y s t a t . see Sec . 4 . 3 f o r a l t . −−
# i n t e g r a t i o n p o i n t s
t = t o r c h . l i n s p a c e ( −5 , 5 , 17 , ∗∗ dev )
# t h e o r e t i c a l CF f o r N( 0 , 1 ) and Gauss . window
exp_f = t o r c h . exp ( −0.5 ∗ t ∗ ∗ 2 )
# e m p i r i c a l CF −− gathered across d e v i c e s −−
x _ t = ( x @ A) . unsqueeze ( 2 ) ∗ t
e c f = ( 1 j ∗ x _ t ) . exp ( ) . mean ( 0 )
e c f = a l l _ r e d u c e ( ecf , op= "AVG" )
# weighted L2 d i s t a n c e
e r r = ( e c f − exp_f ) . abs ( ) . square ( ) . mul ( exp_f )
N = x . s i z e ( 0 ) ∗ w o r l d _ s i z e
T = t o r c h . t r a p z ( e r r ,
r e t u r n T

t , dim =1) ∗ N

# (N, M, T )

for higher moment–assuming they even exist. Even for
well-behaved distributions, raising values to a power of
𝑘 can quickly lead to exploding gradients. This comes in
sharp contrast with the ECF which is always bounded and
with bounded gradients for any input distribution for the
projected samples 𝑧𝑖 = 𝒂⊤ 𝑓𝜃(𝒙𝑛), 𝑛 = 1, . . . , 𝑁.

Theorem 4: Stability of Epps-Pulley Test

(Epps–Pulley) satisfies for samples 𝑧1, . . . , 𝑧𝑁
√

(cid:12)
(cid:12)
(cid:12)
(cid:12)

𝜕𝐸𝑃(a)
𝜕𝑧𝑖

(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤

4𝜎2
𝑁

,

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

𝜕2𝐸𝑃(a)
𝜕𝑧2
𝑖

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤

𝐶

𝜋𝜎3
2𝑁

,

with constant 𝐶, and bandwidth 𝜎. (Proof in Section B.12.)

By the chain rule, thm. 4 directly gives ∥∇𝜃𝐸𝑃(a)∥ ≤
4𝜎2
(cid:205)𝑁
𝑖=1 ∥a⊤∇𝜃 𝑓𝜃(x𝑖)∥, providing stable gradients. The lim-
𝑁
itations of moment-based and CDF-based tests coupled
with thm. 4 justifies our choice of the (Epps–Pulley): (i)
DDP-friendly and scalable, (ii) uniformly bounded gradi-
ents and curvature regardless of input distribution, and
(iii) hyper-parameter free implementation. Lastly, we
highlight that our implementation has a linear memory and
computational complexity of 𝒪(𝑁), with 𝑁 the minibatch size.
The implementation of SIGReg using that statistical test is
provided in algorithm 1, along with computation times of
the forward-backward pass in Table 6.

As a last step before introducing LeJEPA, we ought to

10

Figure 7. Expected directional statistic at the end of training (y-axis)
for varying 𝑀 (number of directions used at each training step, x-axis).
The 𝑀 directions are either resampled (green) or kept fixed (blue) at
each training step. While for fixed directions we benefit from thm. 5
bound where increasing 𝑀 reduces the overall expected loss, being able
to resample at every step provides significant coverage boost for free.

study the requirements on the number of directions (|A|)
for (2) to be effective in high-dimension.

4.3 How SIGReg Beats the Curse of

Dimensionality

This last section seeks to characterize how many slices in A
one must sample for (SIGReg) to be an effective statistical
test. That design is crucial if we hope for LeJEPA to success-
fully converge towards isotropic Gaussian embeddings.

Smoothness Beats the Curse of Dimensionality

Our first argument arguing for a favorable scaling of |A|
with the embedding dimension 𝐾 relies on the smoothness
of 𝑃𝜽 as measured by its Sobolev regularity 𝛼 [Adams
and Fournier, 2003]. We formalize below a bound on
the directional test from Equation (3) over all possible
directions 𝒂 when the test statistic is minimized over
|A| = 𝑀 directions. While we provide bounds on the
expected discrepancy over random directions 𝒂 when the
EP test is satisfied (equals zero) on a finite set of directions,
the provided proof includes the case of moment-based
and CDF-based tests as well.

102103M(log-scale)050010001500Ea(cid:2)T({a>fθ(xn)}Nn=1)(cid:3)β=−2.79R2=0.87β=−285.91R2=0.96randomﬁxedLeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Theorem 5: Unified Error Bounds

Let 𝑝𝜽 ∈ 𝐻 𝛼(R𝐾), 𝒂 ∼ 𝒰 (𝒮 𝐾−1), and (Epps–Pulley)= 0, i.e.,
𝑃(a)
𝜃 = 𝑄(a), ∀𝒂 ∈ A, then

E𝒂

(cid:20)∫

R

2
(cid:12)𝜑𝑎(𝑡) − 𝜑𝒩 (𝑡)(cid:12)
(cid:12)
(cid:12)

𝑑𝑡

(cid:21)

≤ 𝐶(𝐾, 𝛼)|A|−2𝛼/(𝐾−1)

∫ ∞

×

0

2
(cid:13)𝜑·(𝑟) − 𝜑𝒩 (𝑟)(cid:13)
(cid:13)
𝐻 𝛼(𝒮 𝐾−1) 𝑑𝑟,
(cid:13)

(Proof in Section B.10.)

As |A| → ∞, the bound decays as |A|−2𝛼/(𝐾−1), showing
that |A| = 𝑂(𝐾) directions suffice for 𝜖-approximation
when 𝛼 is large. Some examples of embedding densities
with varying 𝛼 are provided in Figure 4. The following
statement characterizes how the 𝑀 directions actually con-
strain the entire space as a function of 𝛼. The constant

2 )
22𝛼𝜋(𝐾−1)/2Γ(𝛼+ 𝐾−1
2 )
(𝐾−1)Γ(𝛼)Γ( 𝐾−1

is visualized in Figure 15 (left)

𝐶(𝐾, 𝛼) =
depicting how 𝛼 and |A| interact. In words, we obtain
that thanks to the natural smoothness of DN–either stem-
ming from the architecture or the implicit and explicit
regularizers used during training–applying SIGReg on |A|
directions can be sufficient to tightly constrain the entire
space. We note that considering the worst case over 𝒂 or
using low-discrepancy sequences for 𝒂 does not impact
the asymptotic bounds, details provided in Section D.

SGD Beats the Curse of Dimensionality

Our second argument leverages the iterative nature of
DN training. Although we may use only |A| to be a few
hundreds, the cumulative number of sampled directions
grows linearly with training time. This resampling effect
(illustrated in Figure 7, bottom) enables rapid convergence.
Even small |A| achieves tight distributional matching com-
pared to keeping the set A fixed throughout minibatches
(recall thm. 5). Our experiments show that even with |A|
as low as 16 can easily outperform a fixed set with |A| of
order of thousands thanks to the compounding effect of
resampling at each minibatch.

Empirical Validation on Synthetic Data

We conclude this section with a controlled experiment
applying (SIGReg) with gradient-based training to pro-
In this setup, we directly
duce isotropic embeddings.
consider embeddings 𝒁 which we will differentiate and
optimized to minimize (SIGReg). By directly optimizing
the embeddings we are able to observe the impact of the
loss without any possible constraint and regularization
that would come from the architecture. We sample 𝑁 i.i.d.
samples 𝒙𝑛 in a 𝐷-dimensional space. This sampling is
based on an isotropic Gaussian distribution–but the first

11

Algorithm 2. LeJEPA implementation–works out-of-the-box on any
dataset, with DDP, with any backbone, e.g., torchvision or timm. For
non-ViT architectures (e.g., ResNet), set global_views = all_views. We
use bs for the minibatch size, SIGReg is from algorithm 1.

def LeJEPA ( g l o b a l _ v i e w s , a l l _ v i e w s ,

lambd ) :

" " " g l o b a l _ v i e w s and a l l _ v i e w s are l i s t s o f

t e n s o r s ,

lambd i s a s c a l a r " " "

# embedding o f g l o b a l views
g_emb = f o r w a r d ( t o r c h . c a t ( glob_views ) )
# embedding o f
# i f
a_emb = f o r w a r d ( t o r c h . c a t ( a l l _ v i e w s ) )

r e s n e t : s k i p w i t h a_emb=g_emb

l o c a l views

# LeJEPA l o s s
c e n t e r s = g_emb . view ( −1 , bs , K) . mean ( 0 )
a_emb = a_emb . view ( −1 , bs , K)
sim = ( c e n t e r s − a_emb ) . square ( ) . mean ( )
s i g r e g = mean ( SIGReg ( emb , g l o b a l _ s t e p )

i n a_emb )

r e t u r n (1 − lambd ) ∗ sim + lambd∗ s i g r e g

f o r emb

two dimensions are again set to the adversarial “X” shape.
That is, among the 𝐷 dimensions, only two must be trans-
formed as all the other ones already obey the isotropic
Gaussian target distribution. We then make the samples
𝒙𝑛 differentiable and optimize then to minimize the value
of the different statistical tests compute on 𝑀 random
𝑀 random directions. Those directions are resampled
after each gradient step–which follows the procedure we
will employ in LeJEPA. We present the results in Figure 6
demonstrating that even in challenging case, i.e., 𝐷 = 512
and 𝑀 = 16, SIGReg is able to detect the two degenerate
dimensions and unfold them back to how they should
look like under the target distribution.

5 LeJEPA: Stable and Scalable

Implementation

Having established that isotropic Gaussians are the optimal
embedding distribution for foundation models (Section 3)
and introduced SIGReg to achieve this distribution (def. 2),
we now present the complete LeJEPA framework. We first
evaluate candidate statistical tests (Sections 4.2.1 and 4.2.2)
and identify characteristic function-based tests as optimal
for gradient-based training (Section 4.2.3). The full LeJEPA
implementation follows in Section 5.1.

5.1 LeJEPA: SIGReg + Prediction Loss

We now discuss the implementation of LeJEPA starting
with SIGReg and followed by the prediction and total
losses.

The SIGReg Loss. We chose (Epps–Pulley) for its prov-
able boundedness (thm. 4) and its scalability. Its implemen-
tation follows exactly the equation except for the integrate
which is estimated using a quadrature approximation. We

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

find that the simple trapezoidal quadrature rule is suffi-
cient even with as few knots as 17, as ablated in Figure 20.
In particular, we leverage the symmetry of the integrand
to double the number of knots for free, see the official code.
On the other hand, the use of minibatches introduces a
bias vanishing at rate 𝒪(1/𝑁), as formalized below.

Theorem 6: Vanishing gradient bias

The expectation of (Epps–Pulley) satisfies

E (cid:104)

(cid:105)

(cid:98)𝐿𝑛(𝜃)

= 𝐿(𝜃) +

1
𝑁

∫

R

𝑤𝑠 (𝑡)(cid:0)1 − |𝜑𝑃(𝑡)|2(cid:1) 𝑑𝑡,

therefore both the loss and its derivative have a bias of order
𝑂(1/𝑛). (Proof in Section B.13.)

Hence, the gradients we obtain from using (Epps–Pulley)
are biased by an explicit 𝒪(1/𝑁) term. We found this bias
to be minimal and not a concern even for minibatches
as small as 16. Unbiased alternatives include using U-
statistic debiasing of |𝜙𝜃|2 or sample splitting, which we
do not explore in this study. Our final implementation of
the SIGReg term with Epps-Pulley statistic is provided in
algorithm 1.

The Prediction Loss. To standardize notations, we adopt
the DINO [Caron et al., 2021] setup of generating 𝑉𝑔 global
views and 𝑉𝑙 local views, leading to a total of 𝑉 = 𝑉𝑔 + 𝑉𝑙
views. We set the first 1, . . . , 𝑉𝑔 indices of each 𝒛𝑛,𝑣 as the
global views. For the cases without local views, simply
set 𝑉𝑙 = 0. The prediction loss is then given by having all
views predict the global views as

ℒpred({𝒛𝑛,𝑣}𝑉

𝑣=1) =

1
𝑉𝑔

𝑉𝑔
(cid:213)

𝑣=1

1
𝑉

𝑉
(cid:213)

𝑣′=1

∥𝒛𝑛,𝑣 − 𝒛𝑛,𝑣′∥2

2

(5)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

1
𝑉g

𝑉g
(cid:213)

𝑣=1

𝒛𝑛,𝑣 − 𝒛𝑛,𝑣′

2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
2

2
(cid:13)𝝁𝑛 − 𝒛𝑛,𝑣′(cid:13)
(cid:13)
(cid:13)
2

,

1
𝑉

=

≜

1
𝑉

𝑉
(cid:213)

𝑣′=1

𝑉
(cid:213)

𝑣′=1

(6)

(7)

where we denote 𝝁𝑛 ≜ 1
𝑉𝑔
Equation (6) derivations are detailed in Section B.6.

𝑣=1 𝒛𝑛,𝑣, the Equation (5) to

(cid:205)𝑉𝑔

LeJEPA Loss. The final total loss simply combines the
above prediction loss along with SIGReg on each views as
per

ℒLeJEPA({𝒙𝑛,𝑣}𝐵,𝑉

𝑛,𝑣=1) =

𝑉
(cid:213)

SIGReg({{𝒛𝑛,𝑣}𝐵

𝑛=1})

𝜆
𝑉

+

1 − 𝜆
𝐵

𝑣=1
𝐵
(cid:213)

𝑛=1

ℒ

(𝑉g)
pred

({𝒛𝑛,𝑣}𝑉

𝑣=1).

(LeJEPA)

12

We present (LeJEPA)’s implementation in algorithm 2.
Altogether, the entire implementation–besides the usual
model definitions, optimizers, and data loaders–only takes
a few dozens lines in PyTorch (algorithms 1 and 2). The
absence of prototypes, stop-gradients, and teacher-student
networks makes (LeJEPA) appealing as it only contains
one hyperparameter, 𝜆, balancing the trade-off between
the prediction and isotropic Gaussian terms.

5.2 Relation to Prior Work
Prior to presenting our experiments (Section 6), we con-
clude by discussing how our proposed LeJEPA and SIGReg
objective relate to existing frameworks in the literature.

While there is no existing solution employing such
slicing and distribution matching for JEPAs, there exists
similar pipelines for generative models and optimal trans-
port. Notably, the Sliced Score Matching [Song et al., 2020]
proposes to leverage univariate slicing of the space to ease
the estimation of a density for generative models. In a
similar vein, the sliced Wasserstein distance [Bonneel et al.,
2015, Nguyen and Ho, 2023] uses such strategy to speed up
and improve optimal transport. Furthermore, when the
integral of the (Epps–Pulley) test is computed exactly, as
opposed to our quadrature, each slice loss value recovers
the kernel MMD [Sriperumbudur et al., 2010, Gretton et al.,
2012, Chwialkowski et al., 2016] measuring the distance be-
tween two distributions–albeit with a quadratic complexity.
Lastly, it is possible to recover some existing SSL frame-
works in the limit by employing LeJEPA with a particular
test–instead of the preferred (Epps–Pulley). For example,
Setting 𝑇({𝑥𝑛}𝐵
𝑛=1) − 1)2
and using that 𝑇 with SIGReg in LeJEPA recovers the
VICReg SSL method in the limit of large number of slices.
In fact, SIGReg will enforce in expectation that E[Z] = 0
and Cov(Z) = I𝑑, where I𝑑 denotes the 𝑑 × 𝑑 identity
matrix–derivations provided in Section B.14. And since
our invariance term is simply the ℓ2 distance between
the views’ embeddings, LeJEPA recovers VICReg for this
degenerate statistical test. Based on thm. 3, we however
strongly advocate against such a setting as it would lead
to shortcut solutions–a phenomenon already observed in
VICReg.

𝑛=1) = mean({𝑥𝑛}𝐵

𝑛=1)2 + (std({𝑥𝑛}𝐵

6 LeJEPA: Empirical Validation
We now use the LeJEPA implementation described in
Section 5.1 to demonstrate its effectiveness through com-
prehensive experiments. We show that LeJEPA: (i) trains
reliably across diverse architectures and datasets (Sec-
tion 6.1), (ii) provides an informative training loss for
model selection (Section 6.2), (iii) outperforms frontier
vision models on small-scale in-domain pretraining (Sec-
tion 6.3), (iv) scales successfully to nearly 1 billion param-
eters on ImageNet-1k (Section 6.4), and (v) learns rich

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Table 1. ViT/Large-14, on inet1k pretraining for 100 epochs and evalu-
ated with frozen backbone linear probing (top1 accuracy, %).LeJEPA’s
performance is stable across all its hyperparameters and while some
may slightly improve performance, e.g., the number of slices |A| and the
projector sizes, none of the choices lead to a catastrophic collapse.

(a) (Epps–Pulley) parameters

integration

num_slices

config/bstat_n_points

[−1, 1]

[−3, 3]

[−5, 5]

512
2048
512
2048
512
2048

5

17

41

71.82
72.88
73.95
75.02
73.71
74.50

72.13
72.30
74.16
74.68
74.21
74.80

72.04
72.69
74.04
74.77
74.15
74.77

(b) Number of local/global views

# global_views (𝑉g)
# views (𝑉 = 𝑉g + 𝑉l)
4
6
8
10

1

2

4

53.06
58.65
64.46
68.97

72.26
73.07
74.24
74.06

–
73.68
73.94
75.08

(c) Mini-batch size

batch_size

128

256

512

1024

72.20

74.15

74.72

74.07

(d) Embedding/Projector dim.

num_slices
emb. dim.
proj. dim.

1024

4096

512

2048

512

2048

64
128
256
512
1024

75.29
74.77
74.56
73.94
73.65

75.32
75.09
74.66
74.11
73.94

75.50
75.26
75.08
74.81
74.71

75.65
75.47
75.02
74.65
74.79

(e) Register tokens

0

1

2

4

8

reg_tokens
num_slices

1024
4096

75.14
75.61

75.18
75.58

75.08
75.67

75.34
75.63

75.23
75.84

moments of the distribution are well-characterized even
with a modest integration range. The number of slices |𝒜|
has a modest effect—while more slices slightly improve
performance, even 512 slices yield competitive results. We
thus recommend to use 17 integration points, an integration
domain of [−5, 5], and 1024 slices as starting points.

13

Figure 8. Inet100 with 400 pretraining epochs and resnet50 backbone.
We depict linear probe performances as a function of 𝜆 and the number of
views 𝑉 (recall (LeJEPA)). We observe that performances are stable over
𝜆–with peak performance obtain by slightly adjust 𝜆 proportionally
to the number of views. The corresponding performance values are
provided in Table 7.

semantic segmentation features without explicit supervi-
sion.

6.1 LeJEPA’s Stability Across

Hyper-Parameters and Architectures

We now demonstrate LeJEPA’s stability across hyperparam-
eters, architectures, and experimental setups. Additional
cross-domain stability results are presented in Section 6.3.
Stability across standard hyperparameters. We begin
by evaluating LeJEPA on ImageNet-100 and ImageNet-
1K. On ImageNet-100, we train a ResNet-50 and vary the
number of views and the loss weighting 𝜆 (Figure 8).
Performance remains stable across both dimensions, lead-
ing us to recommend 𝜆 = 0.05 as a robust default. On
ImageNet-1K, we train a ViT-Large/14 and explore batch
size, as well as the number of global (𝑉g) and local (𝑉l)
views (Table 1b). We find that the configuration commonly
used in prior work (𝑉g = 2, 𝑉l = 8) transfers well to LeJEPA.
Notably, LeJEPA achieves competitive performance with
batch sizes as small as 128 on ImageNet-1K (Table 1c),
suggesting reduced memory requirements compared to
existing methods. We thus recommend to use 𝜆 = 0.05,
𝑉g = 2, 𝑉l = 8, and batch size ≥ 128 as starting points.

Stability across Epps-Pulley hyperparameters. We next
examine hyperparameters specific to LeJEPA: the num-
ber of slices |𝒜| in SIGReg, the integration domain for
the Epps-Pulley test (Epps–Pulley), and the number of
quadrature points for numerical integration. Table 1a
shows ablations on ImageNet-1K with ViT-Large/14. Both
the integration domain and number of quadrature points
have negligible impact on performance. This is expected:
since the characteristic function is accurate at zero, the

10−310−210−1λ(log-scale)74767880828486top1acc.(%)ResNet50-inet100acc.vsλ2Views4Views8ViewsLeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Figure 9. INet10 pretraining and frozen backbone linear evaluation across 50 timm models using LeJEPA out of the box. We cross-validate the
learning rate and weight-decay. While there is a small variation between the best and worst performing model, we clearly see that across 50 models
spanning 8 families, LeJEPA is able to produce non-trivial representations able to solve the downstream task at SOTA levels.

Stability across architectures. A key advantage of
LeJEPA over recent methods (e.g., ĲEPA, DINOv2) is
its architecture-agnostic design. While most modern self-
supervised methods are tailored to Vision Transformers,
LeJEPA works across diverse architecture families with-
out modification. To validate this claim, we pretrain
approximately 50 architectures from 8 different families
on ImageNet-10, selecting all models in the timm library
with fewer than 20M parameters. All models are able to
learn high-quality representations reaching between 91.5%
to 95% top 1 accuracy with frozen backbone linear prob-
ing. It seems that models performing well in supervised
learning setups are also the ones to favor for LeJEPA, such
as resnets and ViTs. We thus recommend to use standard
architectures such as ResNets and ViTs over specialized models
like EfficientNet as stating point.

Removal of popular heuristics. In addition to providing
reliable performance across models and datasets, LeJEPA’s
provable construction enables us to remove many heuristics
traditionally used to prevent collapse. First, prior work
has shown both empirically and theoretically that predic-
tors in image JEPA (without asymmetric information) and
teacher-student architectures serve primarily to prevent
collapse [Grill et al., 2020, Jing et al., 2021, Tian et al.,
2021, Caron et al., 2021, Chen et al., 2021]. Removing
these components produces collapsed encoders, i.e., with
performances at chance-level. Thanks to LeJEPA’s SIGReg
loss, we can remove both the predictor and teacher-student
architecture without suffering from collapse, as shown in
Table 4. While a teacher-student configuration does pro-
vide a small performance boost for ViT models—consistent
with observations in supervised learning via Stochastic

Weight Averaging [Izmailov et al., 2019]—it is not neces-
sary to prevent collapse. In our setup, we apply SWA on
the encoder producing 𝜇 in Equation (6). Second, recent
work demonstrated that register tokens are needed to pre-
vent training instabilities in vision models [Oquab et al.,
2023, Siméoni et al., 2025, Darcet et al., 2023]. We show
in Table 1 that such instabilities likely stem from poorly
conditioned training objectives. In contrast, LeJEPA does
not require register tokens and achieves stable performance
with or without them. We thus recommend training without
a predictor or register tokens, and optionally applying SWA with
ViTs for a possible performance gain.

Experiment Details 1

We strive for simplicity and thus adopt a unified pretraining
pipeline. The following parameters apply to all experiments
and figures unless stated otherwise in the corresponding
caption and come from Section 6.1:

• LeJEPA’s implementation is given in algorithm 2 with

hyperparameter 𝜆

• All backbones are from timm and all optimizers/sched-

ulers are from PyTorch without modifications

• We employ eight views (𝑉 = 8) containing two global
views (𝑉g = 2) with resolution 224x224 and 96x96 for
the local views

• AdamW optimizer with lr ∈ {5𝑒 − 3, 5𝑒 − 4} and wd ∈
{1𝑒 − 1, 1𝑒 − 2, 1𝑒 − 5}–no scheduler on weight-decay,
standard linear warm-up cosine-annealing for lr

6.2 LeJEPA’s Training Loss is Informative

of Downstream Performance

A major challenge in SSL pretraining is the lack of reliable
signals conveying the quality of the learned representa-
tion. As a result, it is common to monitor a supervised

14

2M5M8M10M12M15M18MParameters (Millions)91.592.092.593.093.594.094.595.0Top-1 Accuracy (%)Inet10 – LeJEPA pretrained, frozen backbone, linear eval – 50 architectures (<20M params.)Model Familyconvnextefficientnetinceptionlevitmaxvitmaxxvitresnetvitconvnext˙atto˙olsconvnext˙atto˙rmsconvnext˙femtoconvnext˙nanoconvnext˙nano˙olsconvnext˙pico˙olsconvnext˙zepto˙rmsconvnext˙zepto˙rms˙olsconvnextv2˙femtoefficientnet˙b0˙g8˙gnefficientnet˙b0˙gninception˙next˙attolevit˙128levit˙128slevit˙192levit˙conv˙256maxvit˙nano˙rw˙256maxvit˙pico˙rw˙256maxvit˙rmlp˙nano˙rw˙256maxvit˙rmlp˙pico˙rw˙256resnet14tresnet18dresnet26resnet26dresnet26tresnet32tsresnet33tsresnetblur18resnext26tsvit˙pe˙core˙tiny˙patch16˙384LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Figure 10. (SIGReg, prediction loss) 2𝑑-plane with downstream task accuracy shown with colors from blue (low) to red (high). We clearly observe
that within this plane, there exists trade-off fronts between the two terms of LeJEPA producing similar downstream performance corresponding to
different values of 𝜆. Yet, those fronts are linear and pointed towards the lower left corner, i.e., LeJEPA’s training loss informs of downstream test
performance across models and datasets (columns). Additional models and datasets provided in Figure 21.

where a clear trend with downstream task accuracy can
be observed. More strikingly, the combined training loss
(LeJEPA) with mixing coefficient 𝜆 exhibits very high
Spearman correlation [Spearman, 1961], denoted as 𝜌𝑠, of
about 85% with downstream accuracy–which is consid-
ered a strong signal. This strong relationship holds across
datasets and architectures. As a result, a lower LeJEPA
training loss reliably indicates a better downstream perfor-
mance.

We can further improve this correlation through a sim-
ple scaling law based upon the trade-off weighting hyper-
parameter 𝜆

𝐶(𝛼) = 𝜌𝑠

(cid:18) train_loss
𝜆𝛼

, test_accuracy

(cid:19)

.

(8)

By setting 𝛼 ≈ 0.4, LeJEPA’s training loss is able to achieve
nearly 99% correlation with downstream performance
across multiple datasets and models. We depict the
changes in 𝐶(𝛼) as a function of 𝛼 on multiple datasets
and models in Figure 11, as well as the training LeJEPA
loss against downstream performance in Figure 19. The
strong alignment between LeJEPA’s training loss and
model quality enables label-free SSL model selection
and cross-validation.

6.3 In-Domain LeJEPA Outperforms
Frontier Model Transfer Learning
A key promise of self-supervised learning is to learn uni-
versal representations that generalize across tasks and
domains. However, current frontier foundation models
(e.g., DINOv2/v3, ĲEPA) are pretrained on natural im-
ages forcing practitioners in specialized domains to collect
large amount of labels for supervised finetuning. In fact,
most frontier models can not be trained directly on those
domains as the number of samples may be small and
searching again for the hyper-parameters would be cum-

Figure 11. Spearman correlation (y-axis) between LeJEPA’s training
loss and downstream accuracy on the dataset’s classification task with a
frozen backbone and linear evaluation. The x-axis varies 𝛼 in Equation (8)
following our scaling law of the loss w.r.t. 𝜆. Using 𝛼 = 0 recovers the
plain training loss. We clearly observe a very high correlation already for
𝛼 = 0, which further increases up to 99% for 𝛼 = 0.4. The entire set of
points is obtained across numerous hyper-parameters such as learning
rate, weight decay, number of epochs, 𝜆–demonstrating how LeJEPA’s
training loss is strongly predictive of downstream performance which
can be used for label-free cross-validation.

downstream task performance, sometimes supplemented
with unsupervised embedding statistics [Agrawal et al.,
2022, Garrido et al., 2023, Thilak et al., 2023]. This process
is highly limiting since it requires labeled data that is
costly and overly specialized. This is further exacerbated
in the latest JEPA models where training losses exhibit low
correlation with downstream performance–and may not
even decrease monotonically during training.

In contrast, we find that LeJEPA’s training loss behaves
much more favorably–providing us with a meaningful
signal on model quality. First, we provide in Figure 10,
the 2D plane spanned by the SIGReg and prediction losses

15

100101SIGRegloss(log-scale)10−1Pred.loss(log-scale)resnet50-galaxy1018.9133.9849.0664.1379.21Accuracy100101SIGRegloss(log-scale)10−210−1Pred.loss(log-scale)resnet50-inet1038.5152.3966.2780.1594.03Accuracy100101SIGRegloss(log-scale)10−1100Pred.loss(log-scale)ViT/base-14-inet1k0.2318.3936.5654.7372.90Accuracy−3−2−10123Alignmentcoeﬃcient(α)0.20.40.60.81.0Corr(trainloss/λα,testacc)resnet18ﬂowers102:0.60→0.95resnet50galaxy10:0.85→0.98resnet50inet10:0.81→0.99ViT/base-8inet1k:0.88→0.93ViT/s-8galaxy10:0.88→0.98ViT/s-8inet10:0.90→0.98LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Figure 12. Small architecture in-domain (Galaxy10) LeJEPA pretraining with linear probe evaluation using frozen backbone or full finetuning
(columns) and with varying number of samples per class (x-axis). We compare against state-of-the-art foundation models (DINOv2/v3, ĲEPA) over 3
different random seeds. We observe that LeJEPA enables in-domain pretraining out of the box across architectures and able to outperform frontier
foundation models. Corresponding numbers are provided in Table 3.

Table 2. Few-shot classification accuracy (percentages) on 8 datasets spanning textures, objects, and fine-grained categories. Our LeJEPA achieves
superior performance on fine-grained tasks (DTD, flowers102, food101) while requiring only 100 pretraining epochs compared to I-JEPA’s 300
epochs—a 3× reduction in training time and computational resources without sacrificing downstream task performance. This efficiency gain is
particularly valuable for practical applications where training budget is limited. Bold indicates best performance within the IN-1K comparison group,
all numbers are percentages.

Dataset

shots model

params pretrain epochs DTD aircr.

cars cifar10 cifar100 flowers102 food pets

avg.

LeJEPA ViT-L
304M IN-1K
LeJEPA ConvNeXtV2-H 660M IN-1K
632M IN-1K
I-JEPA ViT-H
632M IN-1K
I-JEPA ViT-H + STOP

100
100
300
300

9.37
33.21
8.07
32.15
9.86
27.71
26.60 11.18

3.40
4.28
4.33
4.75

I-JEPA ViT-H (22K)

632M IN-22K 900

27.98 13.00

3.45

LeJEPA ViT-L
304M IN-1K
LeJEPA ConvNeXtV2-H 660M IN-1K
632M IN-1K
I-JEPA ViT-H
632M IN-1K
I-JEPA ViT-H + STOP

100
100
300
300

64.72 35.25 22.25
61.84 30.67 24.46
57.68 33.82 21.96
57.00 39.77 25.21

I-JEPA ViT-H (22K)

632M IN-22K 900

58.74 43.52 18.27

304M IN-1K
LeJEPA ViT-L
LeJEPA ConvNeXtV2-H 660M IN-1K
632M IN-1K
I-JEPA ViT-H
632M IN-1K
I-JEPA ViT-H + STOP

100
100
300
300

78.30 57.01 57.28
76.60 52.99 54.88
73.32 56.61 54.47
73.87 61.95 61.27

51.65
50.95
56.52
56.27

61.84

85.15
85.74
88.77
90.09

94.83

96.50
96.15
97.54
98.02

27.01
31.48
30.58
35.20

34.70

59.77
63.29
66.42
70.32

75.23

83.71
81.34
86.42
87.78

1

10

all

48.53 17.14 46.11 29.55
48.74 17.95 58.98 31.58
44.69 14.53 53.38 30.20
47.17 15.75 59.47 32.05

89.72 19.62 30.86 35.15

92.53 50.90 77.00 60.95
91.78 49.32 78.53 60.70
88.24 43.97 83.23 60.51
90.16 45.68 85.13 62.92

98.94 49.06 67.66 63.28
91.21 82.05 89.74 79.48
91.11 77.64 89.76 77.56
86.47 81.02 92.11 78.50
88.08 81.72 92.88 80.70

I-JEPA ViT-H (22K)

632M IN-22K 900

75.67 65.39 49.79

98.46

89.95

98.54 81.58 87.19 80.82

Figure 13. Emergent Object Segmentation via Last Layer Thresholding. LeJEPA naturally learns to segment and track salient objects (shown in
attention maps on the right of each video) without explicit supervision. The results display impressive visual quality and strong temporal consistency
across video frames (videos provided on our project page). This emergent capability demonstrates the rich semantic representations learned through our
self-supervised approach.

16

125101001000allSamplesperclass20304050607080Accuracy(%)Fullﬁnetuning125101001000allSamplesperclassFrozenbackboneLeJEPAconvnextv2nano(galaxy10)LeJEPAlevit128(galaxy10)LeJEPAresnet18(galaxy10)LeJEPAresnet34(galaxy10)DINOv2ViT/s(LVD142M)DINOv3ViT/s(LVD1.7B)LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

bersome yet necessary [Assran et al., 2022].

To demonstrate LeJEPA’s versatility and ability to resolve
that current pain-point, we propose to pretrain directly
on a new domain without any change in the loss or the
pretraining pipeline. We select the Galaxy10 dataset, a
galaxy morphology classification task that differs signif-
icantly from natural images in both visual structure and
statistical properties [Balestriero et al., 2025]. The dataset
contains 11,000 training samples across 10 galaxy types.
For LeJEPA, we use the default hyper-parameters and pre-
train for 400 epochs a variety of backbones. We compare
against the latest DINOv2, DINOv3 and ĲEPA. We report
in Figure 12 the top1 accuracy for linear probing both
with frozen backbone and full-finetuning. We observe
that in-domain pretraining with LeJEPA substantially
outperforms state-of-the-art frontier models (DINOv2,
DINOv3) on both linear probing and full finetuning.
Additional datasets and backbones are provided in Table 5
depicting LeJEPA’s ability to train in-domain, even with
a dataset with 1000 samples (flowers102). Coupling this
result with the stability of LeJEPA across architectures and
hyper-parameters should offer a promising alternatives
in domains not yet accounted for by the latest frontier
models.

6.4 LeJEPA Scales Across Data and Models
We now propose to apply LeJEPA over a larger pretrain-
ing dataset, i.e., Imagenet-1k, and over larger backbones
such as ViT/Large (0.3B), ConvNextV2-Huge (0.6B). For
those two models, we reach an online linear probe accu-
racy on inet1k of 77.1% and 78.5% respectively. Beyond
in-distribution performances, we also explore transfer
learning. For those experiments, our baselines are ĲEPA
with a ViT-Huge (0.6B) which is the closest to our setup,
and we also include a recent improved version of ĲEPA
including additional stochastic prediction tasks [Bar et al.,
2023] that is coined ĲEPA + STOP. For LeJEPA, we employ
the same recipe as described in Section 6.1 and report trans-
fer learning performances with frozen backbone in Table 2.
We observe that we consistently outperform ĲEPA while
employed a smaller model and shorted training schedule.
Beyond top1 accuracy, we also echo our findings from
Section 6.2 about LeJEPA’s training loss quality. In our
setup, we observe a very stable and smooth training curve
indicating a stable optimization landscape removing the
need for careful hyperparameter selection (recall thm. 4).
We provide an example on a ViT-gigantic (1.8B parameters)
in Figure 1.

6.5 Emergent Semantic Structure in

LeJEPA Representations

A hallmark of successful self-supervised learning is the
emergence of semantically meaningful attention patterns

17

Figure 14. LeJEPA learns rich semantic representations through
self-supervised learning. PCA visualization of last-layer features from
LeJEPA (ViT-Large, 100 epochs on ImageNet-1K). For each image, fea-
tures are independently projected to RGB using the first 3 principal
components. Without any supervision, LeJEPA spontaneously develops
semantically meaningful representations: notice how warm colors (red/-
magenta/pink) consistently capture foreground objects (parrot bodies,
dog face), while cool colors (cyan/green/yellow) represent backgrounds
and foliage. This emergent object-background separation and percep-
tual grouping discovered the visual structure of the world purely from
unlabeled data.

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

without explicit supervision [Caron et al., 2021]. To as-
sess whether LeJEPA learns such structure, we visualize
the attention maps of the learned representations. Fol-
lowing DINO [Caron et al., 2021], we apply PCA to the
embeddings and visualize the first principal components,
which reveal clear correspondence to object boundaries
and salient regions (Figure 14). Furthermore, we explore
whether these attention patterns can enable unsupervised
video segmentation—a challenging task requiring tempo-
ral consistency and object understanding. By thresholding
the self-attention maps of the [CLS] token, we obtain
binary masks that track objects across frames without
any segmentation labels during training. As shown in Fig-
ure 13, LeJEPA’s attention naturally segments foreground
objects from background with remarkable temporal co-
herence, suggesting that the learned representations cap-
ture both spatial semantics and temporal structure. This
emergent capability demonstrates that LeJEPA’s stability-
focused objective does not sacrifice the semantic richness
of learned features.

7 Conclusion
We have established a principled theoretical framework
for JEPA-based self-supervised learning that fundamen-
tally resolves its core pathologies. Our contributions span
theory and practice: we proved that isotropic Gaussian
embeddings uniquely minimize worst-case downstream
risk, introduced SIGReg as a tractable and provably correct
method to enforce this distribution, and demonstrated
that this approach eliminates representational collapse by
design–and not through ad-hoc combinations of teacher-
student networks, stop-gradients, or asymmetric architec-
tures.

We validate LeJEPA across domains and over 60 archi-
tectures including gigantic versions with 1.8B parameters.
In spite of its simplicify , LeJEPA matches state-of-the-art
performance while requiring fewer than 50 lines of core im-
plementation. Critically, our approach provides what SSL
has long needed: a mathematically rigorous foundation
that directly informs practical algorithm design.

Acknowledgments
We would like to thank Mike Rabbat and Lucas Maes for
providing valuable feedbacks on the manuscript.

References
Haneen Arafat Abu Alfeilat, Ahmad BA Hassanat, Omar
Lasassmeh, Ahmad S Tarawneh, Mahmoud Bashir Al-
hasanat, Hamzeh S Eyal Salman, and VB Surya Prasath.
Effects of distance measure choice on k-nearest neighbor
classifier performance: a review. Big data, 7(4):221–248,
2019.

Robert A Adams and John JF Fournier. Sobolev spaces,

volume 140. Elsevier, 2003.

Kumar K Agrawal, Arnab Kumar Mondal, Arna Ghosh,
and Blake Richards. a-req: Assessing representation
quality in self-supervised learning by measuring eigen-
spectrum decay. Advances in Neural Information Processing
Systems, 35:17626–17638, 2022.

Theodore W Anderson and Donald A Darling. Asymptotic
theory of certain" goodness of fit" criteria based on
stochastic processes. The annals of mathematical statistics,
pages 193–212, 1952.

Mahmoud Assran, Randall Balestriero, Quentin Duval,
Florian Bordes, Ishan Misra, Piotr Bojanowski, Pascal
Vincent, Michael Rabbat, and Nicolas Ballas. The hidden
uniform cluster prior in self-supervised learning. arXiv
preprint arXiv:2210.07277, 2022.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bo-
janowski, Pascal Vincent, Michael Rabbat, Yann LeCun,
and Nicolas Ballas. Self-supervised learning from im-
ages with a joint-embedding predictive architecture. In

Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 15619–15629, 2023.

Randall Balestriero and Yann LeCun. Contrastive and non-
contrastive self-supervised learning recover global and
local spectral embedding methods. Advances in Neural
Information Processing Systems, 35:26671–26685, 2022.

Randall Balestriero and Yann LeCun. Learning by recon-
struction produces uninformative features for percep-
tion. arXiv preprint arXiv:2402.11337, 2024.

Randall Balestriero, Mark Ibrahim, Vlad Sobal, Ari Mor-
cos, Shashank Shekhar, Tom Goldstein, Florian Bordes,
Adrien Bardes, Gregoire Mialon, Yuandong Tian, et al.
A cookbook of self-supervised learning. arXiv preprint
arXiv:2304.12210, 2023.

Randall Balestriero, Nicolas Ballas, Mike Rabbat, and Yann
LeCun. Gaussian embeddings: How jepas secretly learn
your data density. arXiv preprint arXiv:2510.05949, 2025.

Amir Bar, Florian Bordes, Assaf Shocher, Mahmoud As-
sran, Pascal Vincent, Nicolas Ballas, Trevor Darrell,
Amir Globerson, and Yann LeCun. Stochastic posi-
tional embeddings improve masked image modeling.
arXiv preprint arXiv:2308.00566, 2023.

Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg:
Variance-invariance-covariance regularization for self-
supervised learning. arXiv preprint arXiv:2105.04906,
2021.

18

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Jan Beirlant, Edward J Dudewicz, László Györfi, Edward C
Van der Meulen, et al. Nonparametric entropy estima-
tion: An overview. International Journal of Mathematical
and Statistical Sciences, 6(1):17–39, 1997.

Kacper Chwialkowski, Heiko Strathmann, and Arthur
Gretton. A kernel test of goodness of fit. In International
conference on machine learning, pages 2606–2615. PMLR,
2016.

Chris M Bishop. Training with noise is equivalent to
tikhonov regularization. Neural computation, 7(1):108–
116, 1995.

Christopher M Bishop and Nasser M Nasrabadi. Pattern
recognition and machine learning, volume 4. Springer,
2006.

Gunnar Blom. Statistical estimates and transformed beta-

variables. PhD thesis, Almqvist & Wiksell, 1958.

Nicolas Bonneel,

Julien Rabin, Gabriel Peyré, and
Hanspeter Pfister. Sliced and radon wasserstein barycen-
ters of measures. Journal of Mathematical Imaging and
Vision, 51(1):22–45, 2015.

Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard
Säckinger, and Roopak Shah. Signature verification
using a" siamese" time delay neural network. Advances
in neural information processing systems, 6, 1993.

Jerome S Bruner and Leo Postman. On the perception of
incongruity: A paradigm. Journal of personality, 18(2):
206–223, 1949.

Russel E Caflisch. Monte carlo and quasi-monte carlo

methods. Acta numerica, 7:1–49, 1998.

Torsten Carleman. Les Fonctions quasi analytiques: leçons
professées au College de France. Gauthier-Villars, 1926.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jé-
gou, Julien Mairal, Piotr Bojanowski, and Armand Joulin.
Emerging properties in self-supervised vision transform-
ers. In Proceedings of the IEEE/CVF international conference
on computer vision, pages 9650–9660, 2021.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and
Geoffrey Hinton. A simple framework for contrastive
learning of visual representations. In International confer-
ence on machine learning, pages 1597–1607. PmLR, 2020a.

Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad
Norouzi, and Geoffrey E Hinton. Big self-supervised
models are strong semi-supervised learners. Advances
in neural information processing systems, 33:22243–22255,
2020b.

Xinlei Chen, Saining Xie, and Kaiming He. An empirical
study of training self-supervised vision transformers.
In Proceedings of the IEEE/CVF international conference on
computer vision, pages 9640–9649, 2021.

Romain Cosentino, Anirvan Sengupta, Salman Avestimehr,
Mahdi Soltanolkotabi, Antonio Ortega, Ted Willke, and
Mariano Tepper. Toward a geometrical understanding
of self-supervised contrastive learning. arXiv preprint
arXiv:2205.06926, 2022.

Thomas M Cover. Elements of information theory. John Wiley

& Sons, 1999.

Harald Cramér. On the composition of elementary errors:
First paper: Mathematical deductions. Scandinavian
Actuarial Journal, 1928(1):13–74, 1928.

Harald Cramér and Herman Wold. Some theorems on
distribution functions. Journal of the London Mathematical
Society, 1(4):290–294, 1936.

Marco Cuturi, Olivier Teboul, and Jean-Philippe Vert. Dif-
ferentiable ranking and sorting using optimal transport.
Advances in neural information processing systems, 32, 2019.

Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr
Bojanowski. Vision transformers need registers. arXiv
preprint arXiv:2309.16588, 2023.

Josef Dick and Friedrich Pillichshammer. Digital nets
and sequences: discrepancy theory and quasi–Monte Carlo
integration. Cambridge University Press, 2010.

Ted Dunning. The t-digest: Efficient estimates of distribu-

tions. Software Impacts, 7:100049, 2021.

Ted Dunning and Otmar Ertl. Computing extremely

accurate quantiles using t-digests.
arXiv:1902.04023, 2019.

arXiv preprint

Gustav Elfving. The asymptotical distribution of range in
samples from a normal population. Biometrika, 34(1/2):
111–119, 1947.

Thomas W Epps and Lawrence B Pulley. A test for nor-
mality based on the empirical characteristic function.
Biometrika, 70(3):723–726, 1983.

Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto,
and Nicu Sebe. Whitening for self-supervised repre-
sentation learning. In International conference on machine
learning, pages 3015–3024. PMLR, 2021.

David Fan, Shengbang Tong, Jiachen Zhu, Koustuv Sinha,
Zhuang Liu, Xinlei Chen, Michael Rabbat, Nicolas Ballas,
Yann LeCun, Amir Bar, et al. Scaling language-free visual
representation learning. arXiv preprint arXiv:2504.01017,
2025.

19

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Ronald Aylmer Fisher. Statistical methods for research workers.

Number 5. Oliver and Boyd, 1928.

Karl Friston. The free-energy principle: a unified brain
theory? Nature reviews neuroscience, 11(2):127–138, 2010.

Quentin Garrido, Randall Balestriero, Laurent Najman,
and Yann Lecun. Rankme: Assessing the downstream
performance of pretrained self-supervised representa-
tions by their rank. In International conference on machine
learning, pages 10929–10974. PMLR, 2023.

Gene H Golub, Per Christian Hansen, and Dianne P
O’Leary. Tikhonov regularization and total least squares.
SIAM journal on matrix analysis and applications, 21(1):
185–194, 1999.

Ian Goodfellow, Yoshua Bengio, Aaron Courville, and
Yoshua Bengio. Deep learning, volume 1. MIT press
Cambridge, 2016.

Priya Goyal, Dhruv Mahajan, Abhinav Gupta, and Ishan
Misra. Scaling and benchmarking self-supervised visual
In Proceedings of the ieee/cvf
representation learning.
International Conference on computer vision, pages 6391–
6400, 2019.

Richard Langton Gregory. Perceptions as hypotheses.

Philosophical Transactions of the Royal Society of London. B,
Biological Sciences, 290(1038):181–197, 1980.

Arthur Gretton, Karsten M Borgwardt, Malte J Rasch,
Bernhard Schölkopf, and Alexander Smola. A kernel
two-sample test. The journal of machine learning research,
13(1):723–773, 2012.

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin
Tallec, Pierre Richemond, Elena Buchatskaya, Carl Do-
ersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad
Gheshlaghi Azar, et al. Bootstrap your own latent-a new
approach to self-supervised learning. Advances in neural
information processing systems, 33:21271–21284, 2020.

Aditya Grover, Eric Wang, Aaron Zweig, and Stefano
Ermon. Stochastic optimization of sorting networks via
continuous relaxations. arXiv preprint arXiv:1903.08850,
2019.

AK Gupta. Estimation of the mean and standard devia-
tion of a normal population from a censored sample.
Biometrika, 39(3/4):260–273, 1952.

Michael Gutmann and Aapo Hyvärinen. Noise-contrastive
estimation: A new estimation principle for unnormal-
ized statistical models. In Proceedings of the thirteenth
international conference on artificial intelligence and statis-
tics, pages 297–304. JMLR Workshop and Conference
Proceedings, 2010.

JM Hammersley and KW Morton. The estimation of loca-
tion and scale parameters from grouped data. Biometrika,
41(3/4):296–301, 1954.

Felix Hausdorff. Momentprobleme für ein endliches inter-

vall. Mathematische Zeitschrift, 16(1):220–248, 1923.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross
Girshick. Momentum contrast for unsupervised visual
representation learning. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages
9729–9738, 2020.

H von Helmholtz et al. Handbook of physiological optics.

Voss, Leipzig, 1867.

R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon,
Karan Grewal, Phil Bachman, Adam Trischler, and
Yoshua Bengio. Learning deep representations by mu-
tual information estimation and maximization. arXiv
preprint arXiv:1808.06670, 2018.

C. A. R. Hoare. Quicksort. The Computer Journal, 5(1):10–16,
01 1962. ISSN 0010-4620. doi: 10.1093/comjnl/5.1.10.
URL https://doi.org/10.1093/comjnl/5.1.10.

Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov,
Dmitry Vetrov, and Andrew Gordon Wilson. Averaging
weights leads to wider optima and better generalization,
2019. URL https://arxiv.org/abs/1803.05407.

Carlos M Jarque and Anil K Bera. Efficient tests for nor-
mality, homoscedasticity and serial independence of
regression residuals. Economics letters, 6(3):255–259,
1980.

Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian.
Understanding dimensional collapse in contrastive self-
supervised learning. arXiv preprint arXiv:2110.09348,
2021.

Harry Joe. Estimation of entropy and other functionals of
a multivariate density. Annals of the Institute of Statistical
Mathematics, 41(4):683–697, 1989.

Thomas Kerdreux, Alexandre Tuel, Quentin Febvre,
Alexis Mouche, and Bertrand Chapron. Efficient self-
supervised learning for earth observation via dynamic
dataset curation. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 3017–3027, 2025.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ash-
win Balakrishna, Sudeep Dasari, Siddharth Karam-
cheti, Soroush Nasiriany, Mohan Kumar Srirama,
Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid:
A large-scale in-the-wild robot manipulation dataset.
arXiv preprint arXiv:2403.12945, 2024.

20

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Diederik P Kingma, Danilo J Rezende, Shakir Mohamed,
and Max Welling. Semi-supervised learning with deep
generative models. Advances in neural information process-
ing systems, 27, 2014.

A. N. Kolmogorov. Sulla determinazione empirica di una
legge di distribuzione. Giornale dell’Istituto Italiano degli
Attuari, 4:83–91, 1933.

Yann LeCun. A path towards autonomous machine intelli-
gence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62,
2022.

Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep

learning. nature, 521(7553):436–444, 2015.

Erich Leo Lehmann and Joseph P Romano. Testing statistical

hypotheses. Springer, 2005.

Xiao Liu, Fanjin Zhang, Zhenyu Hou, Li Mian, Zhaoyu
Wang, Jing Zhang, and Jie Tang. Self-supervised learn-
ing: Generative or contrastive.
knowledge and data engineering, 35(1):857–876, 2021.

IEEE transactions on

Zhuang Ma and Michael Collins. Noise contrastive esti-
mation and negative sampling for conditional models:
Consistency and statistical efficiency. arXiv preprint
arXiv:1809.01812, 2018.

Tobias Maltenberger, Ivan Ilic, Ilin Tolovski, and Tilmann
Rabl. Evaluating multi-gpu sorting with modern inter-
connects. In Proceedings of the 2022 International Conference
on Management of Data, pages 1795–1809, 2022.

George Marsaglia. Choosing a point from the surface
of a sphere. The Annals of Mathematical Statistics, 43(2):
645–646, 1972.

Charles Masson, Jee E Rim, and Homin K Lee. Ddsketch:
A fast and fully-mergeable quantile sketch with relative-
error guarantees. arXiv preprint arXiv:1908.10693, 2019.

David McAllester and Karl Stratos. Formal limitations on
the measurement of mutual information. In International
Conference on Artificial Intelligence and Statistics, pages
875–884. PMLR, 2020.

H Mhaskar, F Narcowich, and J Ward.

Spheri-
cal marcinkiewicz-zygmund inequalities and positive
quadrature. Mathematics of computation, 70(235):1113–
1130, 2001.

Erik G Miller. A new class of entropy estimators for
multi-dimensional densities. In 2003 IEEE International
Conference on Acoustics, Speech, and Signal Processing, 2003.
Proceedings.(ICASSP’03)., volume 3, pages III–297. IEEE,
2003.

Frederick Mosteller. On some useful “inefficient” statistics.

Springer, 2006.

Elizbar A Nadaraya. On estimating regression. Theory of

Probability & Its Applications, 9(1):141–142, 1964.

Francis J Narcowich, Pencho Petrushev, and Joseph D
Ward. Localized tight frames on spheres. SIAM Journal
on Mathematical Analysis, 38(2):574–594, 2006.

Jerzy Neyman and Egon Sharpe Pearson. Ix. on the prob-
lem of the most efficient tests of statistical hypotheses.

Philosophical Transactions of the Royal Society of London.
Series A, Containing Papers of a Mathematical or Physical
Character, 231(694-706):289–337, 1933.

Khai Nguyen and Nhat Ho. Energy-based sliced wasser-
stein distance. Advances in Neural Information Processing
Systems, 36:18046–18075, 2023.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Rep-
resentation learning with contrastive predictive coding.
arXiv preprint arXiv:1807.03748, 2018.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby,
et al. Dinov2: Learning robust visual features without
supervision. arXiv preprint arXiv:2304.07193, 2023.

Vardan Papyan, XY Han, and David L Donoho. Prevalence
of neural collapse during the terminal phase of deep
learning training. Proceedings of the National Academy of
Sciences, 117(40):24652–24663, 2020.

Felix Petersen, Christian Borgelt, Hilde Kuehne, and Oliver
Deussen. Monotonic differentiable sorting networks.
arXiv preprint arXiv:2203.09630, 2022.

RoL Plackett. Linear estimation from censored data. The
Annals of Mathematical Statistics, 29(1):131–142, 1958.

Ben Poole, Sherjil Ozair, Aaron Van Den Oord, Alex Alemi,
and George Tucker. On variational bounds of mutual in-
formation. In International conference on machine learning,
pages 5171–5180. PMLR, 2019.

M Mahibbur Rahman and Z Govindarajulu. A modifica-
tion of the test of shapiro and wilk for normality. Journal
of Applied Statistics, 24(2):219–236, 1997.

Bryan Rodas, Natalie Montesino, Jakob Ambsdorf, David
Klindt, and Randall Balestriero. Diet-cp: Lightweight
and data efficient self supervised continued pretraining.
arXiv preprint arXiv:2509.06990, 2025.

Samarendra Nath Roy. On a heuristic method of test
construction and its use in multivariate analysis. The
Annals of Mathematical Statistics, 24(2):220–238, 1953.

21

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

David E Rumelhart, Geoffrey E Hinton, and Ronald J
Williams. Learning representations by back-propagating
errors. nature, 323(6088):533–536, 1986.

Claude E Shannon. A mathematical theory of communi-
cation. The Bell system technical journal, 27(3):379–423,
1948.

Samuel S Shapiro and RS Francia. An approximate analysis
of variance test for normality. Journal of the American
statistical Association, 67(337):215–216, 1972.

Samuel Sanford Shapiro and Martin B Wilk. An analy-
sis of variance test for normality (complete samples).
Biometrika, 52(3-4):591–611, 1965.

Ravid Shwartz Ziv and Yann LeCun. To compress or not
to compress—self-supervised learning and information
theory: A review. Entropy, 26(3):252, 2024.

Ravid Shwartz-Ziv, Randall Balestriero, and Yann LeCun.
What do we maximize in self-supervised learning? arXiv
preprint arXiv:2207.10081, 2022.

Bernard W Silverman. Density estimation for statistics and

data analysis. Routledge, 2018.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico
Baldassarre, Maxime Oquab, Cĳo Jose, Vasil Khalidov,
Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa,
et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon.
Sliced score matching: A scalable approach to density
and score estimation. In Uncertainty in artificial intelli-
gence, pages 574–584. PMLR, 2020.

Charles Spearman. The proof and measurement of associ-

ation between two things. 1961.

Bharath K Sriperumbudur, Arthur Gretton, Kenji Fuku-
mizu, Bernhard Schölkopf, and Gert RG Lanckriet.
Hilbert space embeddings and metrics on probability
measures. The Journal of Machine Learning Research, 11:
1517–1561, 2010.

Shiliang Sun and Rongqing Huang. An adaptive k-nearest
neighbor algorithm. In 2010 seventh international confer-
ence on fuzzy systems and knowledge discovery, volume 1,
pages 91–94. IEEE, 2010.

Richard S Sutton. Dyna, an integrated architecture for
learning, planning, and reacting. ACM Sigart Bulletin, 2
(4):160–163, 1991.

Gábor J Székely and Maria L Rizzo. A new test for multi-
variate normality. Journal of Multivariate Analysis, 93(1):
58–80, 2005.

Ivan Tanasic, Lluís Vilanova, Marc Jordà, Javier Cabezas,
Isaac Gelado, Nacho Navarro, and Wen-mei Hwu. Com-
parison based sorting for systems with multiple gpus.
In Proceedings of the 6th Workshop on General Purpose
Processor Using Graphics Processing Units, pages 1–11,
2013.

Kashvi Taunk, Sanjukta De, Srishti Verma, and Aleena Swe-
tapadma. A brief review of nearest neighbor algorithm
for learning and classification. In 2019 international con-
ference on intelligent computing and control systems (ICCS),
pages 1255–1260. IEEE, 2019.

Vimal Thilak, Chen Huang, Omid Saremi, Laurent Dinh,
Hanlin Goh, Preetum Nakkiran, Joshua M Susskind, and
Etai Littwin. Lidar: Sensing linear probing performance
in joint embedding ssl architectures. arXiv preprint
arXiv:2312.04000, 2023.

Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan,
Cordelia Schmid, and Phillip Isola. What makes for
good views for contrastive learning? Advances in neural
information processing systems, 33:6827–6839, 2020.

Yuandong Tian, Xinlei Chen, and Surya Ganguli. Un-
derstanding self-supervised learning dynamics without
contrastive pairs. In International Conference on Machine
Learning, pages 10268–10278. PMLR, 2021.

Edward C Tolman. Cognitive maps in rats and men.

Psychological review, 55(4):189, 1948.

Hugues Van Assel, Mark Ibrahim, Tommaso Biancalani,
Aviv Regev, and Randall Balestriero. Joint embedding
vs reconstruction: Provable benefits of latent space
prediction for self supervised learning. arXiv preprint
arXiv:2505.12477, 2025.

Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua
Bengio, Pierre-Antoine Manzagol, and Léon Bottou.
Stacked denoising autoencoders: Learning useful rep-
resentations in a deep network with a local denoising
criterion. Journal of machine learning research, 11(12), 2010.

Huy V Vo, Vasil Khalidov, Timothée Darcet, Théo
Moutakanni, Nikita Smetanin, Marc Szafraniec, Hugo
Touvron, Camille Couprie, Maxime Oquab, Armand
Joulin, et al. Automatic data curation for self-supervised
learning: A clustering-based approach. arXiv preprint
arXiv:2405.15613, 2024.

Hermann Von Helmholtz. Handbuch der physiologischen

Optik, volume 9. L. Voss, 1867.

Richard Von Mises. Probability, statistics, and truth. Courier

Corporation, 1981.

22

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Xiao Wang, Haoqi Fan, Yuandong Tian, Daisuke Kihara,
and Xinlei Chen. On the importance of asymmetry
for siamese representation learning. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 16570–16579, 2022.

Geoffrey S Watson. Smooth regression analysis. Sankhy¯a:
The Indian Journal of Statistics, Series A, pages 359–372,
1964.

George S Watson. Goodness-of-fit tests on a circle.

Biometrika, 48(1/2):109–114, 1961.

S Weisburg and C Binham. An approximate analysis of
variance test for non-normality suitable for machine
computation. Technometrics, 17:133–134, 1975.

Shichao Zhang, Xuelong Li, Ming Zong, Xiaofeng Zhu, and
Ruili Wang. Efficient knn classification with different
numbers of nearest neighbors. IEEE transactions on neural
networks and learning systems, 29(5):1774–1785, 2017.

Yifan Zhang, Zhiquan Tan, Jingqin Yang, Weiran Huang,
and Yang Yuan. Matrix information theory for self-
supervised learning. arXiv preprint arXiv:2305.17326,
2023.

23

LeJEPA

Appendix

A Additional Details on Nonlinear Probing
A.1 kNN Probing
To allow for more flexible evaluation of the pretrained encoder 𝑓𝜽, it is standard to work with a 𝑘-NN prober [Taunk et al.,
2019], both for regression and classification. We rely on the radial 𝑘-NN variation that leverages a sample-dependent
𝑘–improving performance for non uniform distributions of samples [Sun and Huang, 2010, Zhang et al., 2017, Abu Alfeilat
et al., 2019].

We denote the underlying embedding density as 𝑝𝑧 ∈ 𝐶3 with derivatives of order up to 3 bounded, and finite Fisher
information and covariance. This regularity condition is fulfilled by current encoders. The unknown labels come from
the target function 𝜂 : R𝐾 → R, assumed 𝐶2. We handle classification tasks by setting 𝜂(𝒛) = P(𝑌 = 1 | 𝒛). The training
consists of the 𝑁 embeddings along with their training labels {(𝒛𝑛 , 𝜂(𝒛𝑛))}𝑁
, where we will denote 𝒚𝑛 ≜ 𝜂(𝒛𝑛). The
prediction for a query vector 𝒒 is formed as

𝑛=1

(cid:98)𝒚(𝒒) :=

1
𝒚(𝒒)

(cid:213)

𝒚𝑛 ,

𝑛:∥𝒛𝑛 −𝒒∥≤𝑟0

(kNN)

with 𝒚(𝒒) ≜ #{𝑛 : (cid:13)
(cid:13)
(cid:13) ≤ 𝑟0} counting the number of samples within a 𝑟-radius ball around 𝒒. The radius 𝑟 controls
how many neighbors predictions are averaged to form the query’s prediction. As per the linear probing’s lemma. 1, we
can characterize the bias of the estimator Equation (kNN) at a particular query point, as formalized below.

(cid:13)𝒛𝑛 − 𝒒

Lemma 4: k-NN Pointwise Bias

The (kNN) estimator has bias at query 𝒒 given by

Bias(𝒒) =

𝑟2
0
𝑑 + 2

(cid:16)

∇𝜂(𝒒)⊤∇ log 𝑝𝑧(𝒒) + 1

2 Δ𝜂(𝒛)

(cid:17)

where the remainder 𝑜(𝑟2
0

) is uniform in 𝒒. (Proof in Section B.3.)

To obtain the integrated bias, i.e., over the distribution of query points, we consider the following two properties.
First, the distribution of query points follow the training distribution, i.e., 𝒒 ∼ 𝑝𝑧, second, target function 𝜂 has gradient
which is mean-zero and isotropic with E(cid:2)∇𝜂(𝒛)∇𝜂(𝒛)⊤(cid:3) = 𝜏2
𝑔 ∈ (0, ∞) uniformly in 𝒛. We also have any finite
scalar-constraint on the covariance of the embeddings such as Tr(Σ) = 𝑐 or ∥Σ∥𝐹 = 𝑐 for a finite constant 𝑐.

𝑔 𝐼𝑑 with 𝜏2

+ 𝑜(𝑟2
0 ),

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Theorem 7: k-NN isotropic Gaussian Optimality

The integrated squared bias of (kNN) satisfies

E𝒛

(cid:2)Bias(𝒛)2(cid:3) =

𝑟4
0
(𝐾 + 2)2

𝑔 𝐽(𝑝) + 𝑂(𝑟4
𝜏2
0 ),

and the isotropic Gaussian is the unique minimizer of the integrated square bias. (Proof in Section B.4.)

As a result, we now have a unique minimizer for the optimal embedding density for both the linear and k-NN probes.

A.2 Kernel Probing
As an alternative to (kNN), it is also common to leverage kernel methods, which we consider in this section.

Consider a kernel 𝐾 : R𝐾 → R with the following standard properties

∫

R𝑑

∫

R𝑑

∫

R𝑑

𝐾(𝑢)𝑑𝑢 = 1,

𝑢𝐾(𝑢)𝑑𝑢 = 0,

𝑢𝑢⊤𝐾(𝑢)𝑑𝑢 = 𝜇2(𝐾)𝐼𝑑 ,

𝑅(𝐾) ≜

∫

R𝑑

𝐾(𝑢)2𝑑𝑢 < ∞,

(normalized)

(symmetric)

(isotropic)

(finite roughness)

for some 𝜇2(𝐾) ∈ (0, ∞), some bandwidth ℎ > 0 and denoting 𝐾 ℎ(𝑡) ≜ ℎ−𝑑𝐾(𝑡/ℎ), we remind the reader that the
Nadaraya-Watson estimator, introduced in Nadaraya [1964], Watson [1964], at a query 𝒒 ∈ R𝑑 is

(cid:98)𝒚(𝒒) ≜

(cid:205)𝑁

𝑛=1 𝐾 ℎ(𝒒 − 𝒙𝑛)𝒚𝑛
(cid:205)𝑁
𝑛=1 𝐾 ℎ(𝒒 − 𝒙𝑛)

.

(NW)

Similarly to (kNN), we will see that the performance of (NW) depends crucially on the distribution of the training
points. We have access to our dataset of inputs from 𝑝𝑧 and for each sample 𝒛𝑛 the corresponding target is given
from 𝜂(𝒛𝑛) = E[𝑌𝑛 | 𝒛𝑛]. We also denote the corresponding conditional variance of the target function at that point as
𝑣(𝑥) = Var(𝑌𝑖 | 𝑋𝑖 = 𝑥). We follow the regularity conditions of the k-NN probing derivations and additionally assume
that 𝑝 has sufficiently light tails so that for each coordinate 𝑗, lim∥𝑥∥→∞ 𝑝(𝑥) = 0 and lim∥𝑥∥→∞ 𝑥 𝑗 𝑝(𝑥) = 0. We first derive
the pointwise bias and variance for

(cid:98)𝒚(𝒒).

Lemma 5: Kernel Bias and Variance

For any fixed 𝒒 ∈ R𝑑 with 𝑝(𝒒) > 0, as ℎ → 0 and 𝑛 ℎ𝑑 → ∞,

Bias(cid:2)

Var(cid:2)

(cid:98)𝒚(𝒒)(cid:3) =
(cid:98)𝒚(𝒒)(cid:3) =

ℎ2𝜇2(𝐾)
2
𝑅(𝐾)
𝑛 ℎ𝑑

𝑣(𝒒)
𝑝(𝒒)

+ 𝑜 (cid:0)(𝑛 ℎ𝑑)−1(cid:1) .

(cid:16)

Δ𝒚(𝒒) + 2∇𝒚(𝒒)⊤∇ log 𝑝(𝒒)

(cid:17)

+ 𝑜(ℎ2),

The 𝑜(·) terms are uniform over compact sets where 𝑝 is bounded away from zero. (Proof in Section B.5.)

We now show that, under a fixed mean and total-covariance constraint on 𝑝𝑧, the isotropic Gaussian distribution
uniquely minimizes the bias and variance of the kernel regression estimator at any test point. We restrict the smoothness
class of the target function using

ℳ(𝐿, 𝐵) ≜ (cid:110)

𝑚 ∈ 𝐶2(R𝑑) : ∥∇𝒚(𝒒)∥ ≤ 𝐿,

|Δ𝒚(𝒒)| ≤ 𝐵, ∀𝒒 ∈ R𝑑(cid:111)

,

25

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

allowing us to formalize below the worst case integrated bias and the optimal density for 𝑧.

Theorem 8: Kernel isotropic Gaussian Optimality

The integrated squared bias of (NW) satisfies

sup
𝑚∈ℳ(𝐿,𝐵)

E𝑧 (cid:2)Bias(cid:2)

(cid:98)𝒚(𝒛)(cid:3) (cid:3)

≤

(cid:16) ℎ2𝜇2(𝐾)
2

(cid:17) 2 (cid:16)

2𝐵2

+

8𝐿2𝐽(𝑝)

(cid:17)

+

𝑜(ℎ4),

and the integrated variance is independent of 𝑝. Among all densities 𝑝 on R𝑑 with total-variance constrained, e.g., Tr(Σ) = 𝑐, the
isotropic Gaussian is the unique minimizer. (Proof in Section B.7.)

B Proofs
B.1 Proof of lemma. 1
Proof. Our proof follows standard derivations when it comes to studying the bias of an estimator. Let’s consider the
ridge regression problem (Tikhonov regularized least squares estimator) with close form estimator

ˆ𝜷 = (X𝑇X + 𝜆wdI)−1X𝑇Y.

(9)

The labels are formed from the ground truth parameter 𝛽true with centered error, as per Y = X𝜷true + 𝜺 where E[𝜺] = 0.
We can now look at the bias of our estimator given by

Bias( ˆ𝜷) = E[ ˆ𝜷] − 𝜷true

= (X𝑇X + 𝜆wdI)−1X𝑇X𝜷true − 𝜷true
= −𝜆wd(X𝑇X + 𝜆wdI)−1𝜷true
= −𝜆wdQ(𝚲 + 𝜆I)−1Q𝑇 𝜷true

We will now compare that bias when 𝑿 has isotropic and anisotropic covariance with same total variance:

𝜆1 + 𝜆2 + · · · + 𝜆𝑝
𝑝

= ¯𝜆.

(10)

For any anisotropic covariance matrix of 𝑿 , denote by 𝒒1 the eigenvector with smallest eigenvalue, and let’s denote by
𝜅 > 0 a positive constant. We now define

𝜷true = 𝜅 · q𝑝 ,

(11)

leading to

∥Bias( ˆ𝜷)∥isotropic =

∥Bias( ˆ𝜷)∥non-isotropic =

𝜆wd
¯𝜆 + 𝜆wd
𝜆wd
𝜆𝑝 + 𝜆wd

∥𝜷true∥,

∥𝜷true∥

Since 𝜆𝑝 < ¯𝜆 (strict inequality when not isotropic):

we obtain that

𝜆wd
𝜆𝑝 + 𝜆wd

>

𝜆wd
¯𝜆 + 𝜆wd

∥Bias( ˆ𝜷)∥non-isotropic > ∥Bias( ˆ𝜷)∥isotropic

As a result, whenever the covariance matrix of 𝑿 is anisotropic, there will be downstream tasks for which the estimator
bias is increased compared to having isotropic covariance matrix. Anisotropic covariance structure thus amplifies
□
regularization bias when the true parameter vector aligns unfavorably with the data’s covariance structure.

26

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

B.2 Proof of lemma. 2
Proof. We use the same formula as in Section B.1 with 𝜆wd = 0. We first see that the estimator is unbiased. We will now
leverage that result to compute the covariance matrix of the estimator

Var( ˆ𝜷|X) = E[( ˆ𝜷 − 𝜷)( ˆ𝜷 − 𝜷)𝑇|X]

= E[(X𝑇X)−1X𝑇 𝜺𝜺𝑇X(X𝑇X)−1|X]
= (X𝑇X)−1X𝑇E[𝜺𝜺𝑇|X]X(X𝑇X)−1
= (X𝑇X)−1X𝑇(𝜎2I𝑛)X(X𝑇X)−1
= 𝜎2(X𝑇X)−1

leading to the total variance

where we used the eigendecomposition:

tr(Var( ˆ𝜷)) = 𝜎2tr(G−1) = 𝜎2

𝑝
(cid:213)

𝑗=1

1
𝜆𝑗

G = Q𝚲Q𝑇

The function 𝑓 (𝑥) = 1

𝑥 is strictly convex on (0, ∞) allowing us to leverage Jensen’s Inequality:

⇐⇒

1
𝐾

𝐾
(cid:213)

𝑘=1

1
𝜆𝑘

>

1
𝐾

1
𝐾

𝐾
(cid:213)

𝑘=1

1
𝜆𝑘

>

1
(cid:205)𝐾
𝑗=1 𝜆𝑘

1
(cid:205)𝐾
𝑗=1 𝜆𝑘

1
𝐾

1
𝐾

𝐾
(cid:213)

𝑘=1
𝐾
(cid:213)

𝐾
(cid:213)

⇐⇒

1
1
𝜆𝑘
(cid:205)𝐾
𝑗=1 𝜆𝑘
⇐⇒ tr(Var( ˆ𝜷))aniso > tr(Var( ˆ𝜷))iso

𝑘=1

𝑘=1

1
𝐾

>

The inequality is strict whenever the eigenvalues {𝜆𝑗}𝑝

𝑗=1

are not all equal.

□

B.3 Proof of lemma. 4
Proof. Under PPP, conditional expectations of

𝜂(𝑥) coincide with the normalized ball average
(cid:98)

E(cid:2)

𝜂(𝑥)(cid:3) =
(cid:98)

∫
B(0,𝑟0)
∫
B(0,𝑟0)

𝜂(𝑥 + 𝑧)𝑝(𝑥 + 𝑧)𝑑𝑧

𝑝(𝑥 + 𝑧)𝑑𝑧

to second order in 𝑟0,

which is the key surrogate used below. Ball integrals. For computations we use (by symmetry) for any 𝑟 > 0:

∫

B(0,𝑟)

𝑧𝑑𝑧 = 0,

∫

B(0,𝑟)

𝑧𝑧⊤𝑑𝑧 =

Vol𝑑+2
𝑑 + 2

𝐼𝑑 ,

∫

B(0,𝑟)

∥𝑧∥2 𝑑𝑧 =

𝑑Vol𝑑+2
𝑑 + 2

.

Fix 𝑥 ∈ R𝑑 and write 𝑧 ∈ B(0, 𝑟0) for local displacements. Assume 𝑝 ∈ 𝐶3, 𝜂 ∈ 𝐶2 with bounded derivatives on the

region of interest, and expand a second-order Taylor expansion:

𝑝(𝑥 + 𝑧) = 𝑝(𝑥) + ∇𝑝(𝑥)⊤𝑧 + 1
𝜂(𝑥 + 𝑧) = 𝜂(𝑥) + ∇𝜂(𝑥)⊤𝑧 + 1

2 𝑧⊤𝐻𝑝(𝑥)𝑧 + 𝑂(∥𝑧∥3),
2 𝑧⊤𝐻𝜂(𝑥)𝑧 + 𝑂(∥𝑧∥3),

with remainders satisfying |𝑅𝜂(𝑥; 𝑧)| ≤ 𝐶𝜂 ∥𝑧∥3 and |𝑅𝑝(𝑥; 𝑧)| ≤ 𝐶𝑝 ∥𝑧∥3 uniformly for ∥𝑧∥ ≤ 𝑟0. Using the ball identities

27

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

∫
𝐵(0,𝑟)

𝑧𝑑𝑧 = 0 and ∫

𝐵(0,𝑟)

𝑧𝑧⊤𝑑𝑧 =

𝑣𝑑 𝑟 𝑑+2
𝑑+2 𝐼𝑑 and collecting terms up to order 𝑟 𝑑+2

0

, we simplify the denominator as

∫

𝒟(𝑥) ≜

𝑝(𝑥 + 𝑧)𝑑𝑧

B(0,𝑟0)

(cid:104)

∫

=

B(0,𝑟0)

𝑝(𝑥) + ∇𝑝(𝑥)⊤𝑧 + 1

2 𝑧⊤𝐻𝑝(𝑥)𝑧 + 𝑅𝑝(𝑥; 𝑧)

(cid:105)

𝑑𝑧

= Vol𝑑

0 𝑝(𝑥) +

Vol𝑑+2
0
2(𝑑 + 2)

tr(cid:0)𝐻𝑝(𝑥)(cid:1) + 𝑂(𝑟 𝑑+3

0

),

since ∫ 𝑧𝑑𝑧 = 0 and ∫ 𝑧⊤𝐻𝑝𝑧𝑑𝑧 = tr(𝐻𝑝)

𝑣𝑑 𝑟 𝑑+2
𝑑+2 and the denominator as
0

∫

𝒩 (𝑥) ≜

B(0,𝑟0)

𝜂(𝑥 + 𝑧)𝑝(𝑥 + 𝑧)𝑑𝑧

∫ (cid:104)

=

𝜂(𝑥) + ∇𝜂(𝑥)⊤𝑧 + 1

2 𝑧⊤𝐻𝜂(𝑥)𝑧

(cid:105) (cid:104)

𝑝(𝑥) + ∇𝑝(𝑥)⊤𝑧 + 1

2 𝑧⊤𝐻𝑝(𝑥)𝑧

(cid:105)

𝑑𝑧 + 𝑂(𝑟 𝑑+3

0

)

= 𝜂(𝑥)𝑝(𝑥)𝑣𝑑𝑟 𝑑

0 + 𝜂(𝑥)

𝑣𝑑𝑟 𝑑+2
0
2(𝑑 + 2)

tr(cid:0)𝐻𝑝(𝑥)(cid:1) +

𝑣𝑑𝑟 𝑑+2
0
𝑑 + 2

∇𝜂(𝑥) · ∇𝑝(𝑥) +

𝑣𝑑𝑟 𝑑+2
0
2(𝑑 + 2)

𝑝(𝑥)tr(cid:0)𝐻𝜂(𝑥)(cid:1) + 𝑂(𝑟 𝑑+3

0

).

Cubic terms vanish by symmetry, and quartic terms are 𝑂(𝑟 𝑑+4

0

). Subtract 𝜂(𝑥)𝒟(𝑥) to obtain the bias numerator:

𝒩 (𝑥) − 𝜂(𝑥)𝒟(𝑥) =

𝑣𝑑𝑟 𝑑+2
0
𝑑 + 2

(cid:16)

∇𝜂(𝑥) · ∇𝑝(𝑥) + 1

2 𝑝(𝑥)Δ𝜂(𝑥)

(cid:17)

+ 𝑂(𝑟 𝑑+3

0

).

Write 𝒟(𝑥) = 𝑣𝑑𝑟 𝑑

0 𝑝(𝑥)(cid:0)1 + 𝛼(𝑥)𝑟2

0 + 𝑂(𝑟3

0)(cid:1) where 𝛼(𝑥) :=

1
2(𝑑+2)𝑝(𝑥)

tr(𝐻𝑝(𝑥)). Then

𝒩 (𝑥)
𝒟(𝑥)

− 𝜂(𝑥) =

=

=

𝑣𝑑 𝑟 𝑑+2
0
𝑑+2

𝑟2
0
𝑑 + 2

)

0

(cid:0)∇𝜂 · ∇𝑝 + 1
2 𝑝Δ𝜂(cid:1) + 𝑂(𝑟 𝑑+3
0 )(cid:1)
0 𝑝 (cid:0)1 + 𝛼𝑟2
0 + 𝑂(𝑟3
𝑣𝑑𝑟 𝑑
(cid:19) (cid:16)
1
2

(cid:18) ∇𝜂 · ∇𝑝
𝑝

1 − 𝛼𝑟2

Δ𝜂

+

0 + 𝑂(𝑟3
0 )

(cid:17)

+ 𝑂(𝑟3
0 )

(cid:16)

𝑟2
0
𝑑 + 2

∇𝜂(𝑥) · ∇ log 𝑝(𝑥) + 1

2 Δ𝜂(𝑥)

(cid:17)

+ 𝑜(𝑟2

0),

uniformly on 𝒦 . This gives the bias formula

E(cid:2)

𝜂(𝑥)(cid:3) − 𝜂(𝑥) =
(cid:98)

(cid:16)

𝑟2
0
𝑑 + 2

∇𝜂(𝑥) · ∇ log 𝑝(𝑥) + 1

2 Δ𝜂(𝑥)

(cid:17)

+ 𝑜(𝑟2

0 ),

completing the proof.

□

B.4 Proof of thm. 7
Proof. Recall from Section B.3 that the bias term as sample 𝒙 is given by

Bias(𝒙) =

=

𝑟2
0
𝑑 + 2
𝑟2
0
𝑑 + 2

(cid:16)

∇𝜂(𝑥) · ∇ log 𝑝(𝑥)

(cid:17)

+

𝑟2
0
2(𝑑 + 2)

Δ𝜂(𝑥) + 𝑜(𝑟2
0 )

(cid:0)𝐴(𝑥) + 𝐶(𝑥)(cid:1) + 𝑜(𝑟2

0 ),

28

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

where we defined 𝐴(𝑥) ≜ ∇𝜂(𝑥) · ∇ log 𝑝(𝑥) and 𝐶(𝑥) ≜ 1
isotropic gradient prior

2 Δ𝜂(𝑥). We now square and take expectation of 𝑋 ∼ 𝑝 and the

E(cid:2)Bias(𝑋)2(cid:3) = E(cid:2)

(cid:33) 2

(cid:32)

𝑟2
0
𝑑 + 2

0 )(cid:3)
(cid:0)𝐴(𝑥)2 + 2𝐴(𝑥)𝐶(𝑥) + 𝐶(𝑥)2(cid:1) + 𝑜(𝑟4

(cid:33) 2

(cid:32)

𝑟2
0
𝑑 + 2

=

(cid:110) E(cid:2)𝐴(𝑋)2(cid:3)
(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)
(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)
(cid:125)

(cid:123)(cid:122)
score-gradient term

(cid:124)

+ 2E(cid:2)𝐴(𝑋)𝐶(𝑋)(cid:3)
(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)
(cid:123)(cid:122)
(cid:125)
cross term

(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)

(cid:124)

(cid:111)

+ 𝑜(𝑟4

0).

+ E(cid:2)𝐶(𝑋)2(cid:3)
(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)
(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)
(cid:124)
(cid:125)
(cid:123)(cid:122)
curvature term

(12)

(13)

We will derive each term separately, recalling that we assume an isotropic gradient prior for 𝜂, i.e., E(cid:2)∇𝜂(𝑥)(cid:3) = 0 and
E(cid:2)∇𝜂(𝑥)∇𝜂(𝑥)⊤(cid:3) = 𝜏2

𝑔 𝐼𝑑, for some 𝜏2

𝑔 ∈ (0, ∞).

1) The score-gradient term E[𝐴(𝑋)2]. Using 𝑣(𝑥) := ∇ log 𝑝(𝑥) for brevity:

(cid:2)E𝜂[𝐴(𝑋)2](cid:3)
(cid:2)E𝜂[(cid:0)∇𝜂(𝑥)⊤𝑣(𝑥)(cid:1) 2](cid:3)
(cid:2)E𝜂[∇𝜂(𝑥)⊤ (cid:16)
(cid:16)

𝑣(𝑥)𝑣(𝑥)⊤(cid:17)
∇𝜂(𝑥)](cid:3)
𝑣(𝑥)𝑣(𝑥)⊤∇𝜂(𝑥)∇𝜂(𝑥)⊤(cid:17)

E(cid:2)𝐴(𝑋)2(cid:3) =E𝑋
=E𝑋

=E𝑋

=E𝑋

=E𝑋

(cid:2)E𝜂[tr
(cid:16)

(cid:2)tr

](cid:3)

(cid:17)(cid:3)

𝑣(𝑥)𝑣(𝑥)⊤E𝜂[∇𝜂(𝑥)∇𝜂(𝑥)⊤]
𝑔∥𝑣(𝑥)∥2(cid:3)
(cid:2)∥𝑣(𝑋)∥2(cid:3)

∥∇ log 𝑝(𝑥)∥2𝑝(𝑥)𝑑𝑥

(cid:2)𝜏2

=E𝑋
=𝜏2

=𝜏2
𝑔

𝑔E𝑋
∫

R𝑑

recovering the Fisher-information functional 𝐽(𝑝), scaled by 𝜏2
𝑔

2) The cross term 2E[𝐴(𝑋)𝐶(𝑋)]. We have

𝐴(𝑥)𝐶(𝑥) =

1
2

(cid:0)∇𝜂(𝑥)⊤𝑣(𝑥)(cid:1)Δ𝜂(𝑥).

Under the prior, ∇𝜂 is mean-zero and isotropic; if, additionally, Δ𝜂 is uncorrelated with ∇𝜂 and has zero mean (or is
bounded and mean-zero after centering), then E𝜂[𝐴(𝑥)𝐶(𝑥)] = 0. If one does not assume the orthogonality/vanishing
covariance above, then E[𝐴(𝑋)𝐶(𝑋)] is a finite constant (depending on the joint law of derivatives of 𝜂), and the cross
term contributes

(cid:33) 2

(cid:32)

𝑟2
0
𝑑 + 2

· 2E[𝐴(𝑋)𝐶(𝑋)] = 𝑂(𝑟4

0 ),

not 𝑜(𝑟4

0 ). In that general case, the leading 𝑝-dependent term of E[Bias(𝑋)2] is still the score-gradient 𝜏2

𝑔 𝐽(𝑝).

3) The curvature term E[𝐶(𝑋)2].

E(cid:2)𝐶(𝑋)2(cid:3) =E𝑋
1
4

=

(cid:2)E𝜂[𝐶(𝑋)2](cid:3)

E𝑋

(cid:2)E𝜂[(Δ𝜂(𝑋))2(cid:3)

which is independent of 𝑝, hence E(cid:2)𝐶(𝑋)2(cid:3) = 𝑂(1)

29

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Putting it together. Substituting into (13):

E(cid:2)Bias(𝑋)2(cid:3) =

(cid:33) 2

(cid:110)

(cid:32)

𝑟2
0
𝑑 + 2

𝜏2
𝑔 𝐽(𝑝) + 𝑂(1)

(cid:111)

+ 𝑜(𝑟4
0)

𝑟4
0
(𝑑 + 2)2

=

𝑔 𝐽(𝑝) + 𝑂(𝑟4
𝜏2
0),

We show that, among all mean-zero distributions 𝑝 on R𝑑 with a given scalar constraint on the covariance (trace,

determinant, Frobenius norm, or spectral radius), the density that minimizes the Fisher-information functional

𝐽(𝑝) :=

∫

R𝑑

∥∇ log 𝑝(𝑥)∥2𝑝(𝑥)𝑑𝑥

is the Gaussian with isotropic covariance satisfying the same scalar constraint. We proceed in two steps: (i) for fixed
covariance matrix Σ ≻ 0, 𝐽(𝑝) is minimized by the Gaussian 𝒩 (0, Σ) and attains the value tr(Σ−1); (ii) for each scalar
constraint, tr(Σ−1) is minimized by Σ = 𝑠𝐼𝑑 for the appropriate scalar 𝑠 > 0.

Lemma 6: Special case: Recovery of VCReg

Let 𝑝 be a mean-zero probability density on R𝑑 with covariance Σ = E[𝑋𝑋⊤] ≻ 0. Then

𝐽(𝑝) ≥ tr(Σ−1),

with equality if and only if 𝑝 = 𝒩 (0, Σ).

Proof. Consider the location family 𝑝𝜃(𝑥) := 𝑝(𝑥 − 𝜃), 𝜃 ∈ R𝑑. Its Fisher-information matrix at 𝜃 is

ℐ (𝜃) = E(cid:2)∇𝜃 log 𝑝𝜃(𝑋)∇𝜃 log 𝑝𝜃(𝑋)⊤(cid:3) = E(cid:2)∇ log 𝑝(𝑋)∇ log 𝑝(𝑋)⊤(cid:3) ,

so that 𝐽(𝑝) = trℐ (𝜃). The estimator 𝑇(𝑋) ≡ 𝑋 is unbiased for 𝜃 under 𝑝𝜃, with Cov(𝑇) = Σ. The matrix Cramér–Rao
bound gives Cov(𝑇) ⪰ ℐ (𝜃)−1, i.e., ℐ (𝜃) ⪰ Σ−1. Taking traces yields 𝐽(𝑝) ≥ tr(Σ−1). Equality in the matrix Cramér–Rao
bound holds if and only if the score is an affine function of 𝑋 − 𝜃, i.e., ∇ log 𝑝𝜃(𝑋) = 𝐴(𝑋 − 𝜃) a.s. for some matrix 𝐴;
□
integrating this identity shows 𝑝𝜃 is Gaussian with precision matrix −𝐴, hence 𝑝 = 𝒩 (0, Σ).

Step 2: Optimizing over covariance shapes under scalar constraints
Write the eigenvalues of Σ as 𝜆1, . . . , 𝜆𝑑 > 0. Then

tr(Σ−1) =

𝑑
(cid:213)

𝑖=1

1
𝜆𝑖

.

We now solve min (cid:205)𝑖 1/𝜆𝑖 under each scalar constraint; in every case the minimum is attained when all 𝜆𝑖 are equal, i.e.,
Σ = 𝑠𝐼𝑑.

(a) Trace constraint. Given tr(Σ) = (cid:205)𝑖 𝜆𝑖 = 𝑡 > 0, by Cauchy–Schwarz,

(cid:32) 𝑑
(cid:213)

𝑖=1

1
𝜆𝑖

(cid:33) (cid:32) 𝑑
(cid:213)

(cid:33)

𝜆𝑖

≥

𝑖=1

(cid:32) 𝑑
(cid:213)

𝑖=1

(cid:33) 2

1

= 𝑑2,

with equality if and only if 𝜆1 = · · · = 𝜆𝑑. Hence

min
Σ≻0: tr(Σ)=𝑡

tr(Σ−1) =

𝑑2
𝑡

,

attained at Σ =

𝑡
𝑑

𝐼𝑑.

30

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

(b) Determinant constraint. Given det(Σ) = (cid:206)𝑖 𝜆𝑖 = 𝛿 > 0, set 𝜇𝑖
inequality,

:= 1/𝜆𝑖 so that (cid:206)𝑖 𝜇𝑖 = 𝛿−1. By the AM–GM

1
𝑑

𝑑
(cid:213)

𝑖=1

𝜇𝑖 ≥

(cid:33) 1/𝑑

(cid:32) 𝑑
(cid:214)

𝑖=1

𝜇𝑖

= 𝛿−1/𝑑 ,

with equality iff 𝜇1 = · · · = 𝜇𝑑, i.e., 𝜆1 = · · · = 𝜆𝑑. Thus

min
Σ≻0: det(Σ)=𝛿

tr(Σ−1) = 𝑑𝛿−1/𝑑 ,

attained at Σ = 𝛿1/𝑑𝐼𝑑.

(c) Frobenius-norm constraint. Given ∥Σ∥2
𝑔(𝜆) := (cid:205)𝑖 𝜆2
𝑖 = 𝑐2. The Lagrangian

𝐹 = (cid:205)𝑖 𝜆2

𝑖 = 𝑐2 > 0, minimize 𝑓 (𝜆) := (cid:205)𝑖 1/𝜆𝑖 over 𝜆𝑖 > 0 subject to

𝑑
(cid:213)

ℒ(𝜆, 𝜈) =

1
𝜆𝑖

+ 𝜈

(cid:32) 𝑑
(cid:213)

𝑖=1

(cid:33)

𝑖 − 𝑐2
𝜆2

𝑖=1
+ 2𝜈𝜆𝑖 = 0 for all 𝑖, i.e., 𝜆3

𝑖 = 1

2𝜈 , so all 𝜆𝑖 are equal. Imposing (cid:205) 𝜆2

𝑖 = 𝑐2 yields 𝜆𝑖 = 𝑐/

√

𝑑,

has first-order conditions −𝜆−2
hence

𝑖

min
Σ≻0: ∥Σ∥𝐹=𝑐

tr(Σ−1) =

𝑑
(cid:213)

𝑖=1

1
𝜆𝑖

=

𝑑3/2
𝑐

,

attained at Σ =

𝑐
√

𝑑

𝐼𝑑.

(d) Spectral-radius constraint. Let the spectral radius be constrained by 𝜌(Σ) = max𝑖 𝜆𝑖 ≤ 𝑟 for some 𝑟 > 0. Since
𝑥 ↦→ 1/𝑥 is strictly decreasing on (0, ∞),

𝑖=1
with equality if and only if 𝜆𝑖 = 𝑟 for all 𝑖. Therefore

𝑑
(cid:213)

1
𝜆𝑖

≥

𝑑
(cid:213)

𝑖=1

1
𝑟

=

𝑑
𝑟

,

min
Σ≻0: 𝜌(Σ)≤𝑟

tr(Σ−1) =

𝑑
𝑟

,

attained at Σ = 𝑟𝐼𝑑.

(The same conclusion holds if the constraint is 𝜌(Σ) = 𝑟, since one may take all eigenvalues equal to 𝑟.)

Conclusion: Isotropic Gaussian is optimal
Combining Lemma 6 with the solutions (a)–(d), we obtain:

31

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Theorem 9: Special case: Recovery of VCReg

Fix one of the following scalar covariance constraints for a mean-zero distribution 𝑝 on R𝑑:

• trace: tr(Cov(𝑋)) = 𝑡,
• determinant: det(Cov(𝑋)) = 𝛿,
• Frobenius norm: ∥Cov(𝑋)∥𝐹 = 𝑐,
• spectral radius upper bound: 𝜌(Cov(𝑋)) ≤ 𝑟.

Then the Fisher-information functional 𝐽(𝑝) is minimized over all such 𝑝 by the isotropic Gaussian 𝑝𝐺 = 𝒩 (0, 𝑠𝐼𝑑) with 𝑠 chosen to
satisfy the constraint. The minimal values are:

trace 𝑡 :

determinant 𝛿 :

𝑑2
𝐽min =
𝑡
𝐽min = 𝑑𝛿−1/𝑑 ,

,

𝑠 =

,

𝑡
𝑑
𝑠 = 𝛿1/𝑑 ,

Frobenius 𝑐 :

𝐽min =

spectral radius 𝑟 :

𝐽min =

In each case, 𝑝𝐺 is the unique minimizer (up to null sets).

𝑑3/2
𝑐

,

𝑑
𝑟

,

𝑠 =

𝑐
√

𝑑

,

𝑠 = 𝑟.

Proof. For any admissible 𝑝 with covariance Σ, Lemma 6 gives 𝐽(𝑝) ≥ tr(Σ−1). Minimizing the right-hand side under
the stated scalar constraint yields Σ = 𝑠𝐼𝑑 by the calculations in (a)–(d). Equality in Lemma 6 holds if and only if 𝑝 is
□
Gaussian with that covariance, hence 𝑝𝐺 uniquely attains the bound.

□

B.5 Proof of lemma. 5
Proof. Write the numerator and denominator of

𝑚(𝑥) as
(cid:98)

𝐵𝑛(𝑥) :=

𝑛
(cid:213)

𝑖=1

𝐾 ℎ(𝑥 − 𝑋𝑖)𝑌𝑖 ,

𝐴𝑛(𝑥) :=

𝑛
(cid:213)

𝑖=1

𝐾 ℎ(𝑥 − 𝑋𝑖),

so that

𝑚(𝑥) =
(cid:98)

𝐵𝑛 (𝑥)
𝐴𝑛 (𝑥)

. Bias. Compute expectations using independence and change of variables. For the denominator,

E[𝐴𝑛(𝑥)] = 𝑛E(cid:2)𝐾 ℎ(𝑥 − 𝑋)(cid:3)
∫

ℎ−𝑑𝐾

(cid:16) 𝑥 − 𝑢
ℎ

(cid:17)

𝑝(𝑢)𝑑𝑢

= 𝑛

= 𝑛

= 𝑛

R𝑑

∫

R𝑑

∫

R𝑑

𝐾(𝑡)𝑝(𝑥 − ℎ𝑡)𝑑𝑡

(𝑡 := (𝑥 − 𝑢)/ℎ)

(cid:16)

𝐾(𝑡)

𝑝(𝑥) − ℎ𝑡⊤∇𝑝(𝑥) +

𝑡⊤∇2𝑝(𝑥)𝑡 + 𝑜(ℎ2)

(cid:17)

𝑑𝑡

ℎ2
2

(cid:16)

= 𝑛

𝑝(𝑥) +

ℎ2
2

∫

(cid:124)

𝑡⊤∇2𝑝(𝑥)𝑡𝐾(𝑡)𝑑𝑡
(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)

(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)

(cid:123)(cid:122)
=𝜇2(𝐾)Δ𝑝(𝑥)

(cid:125)

+𝑜(ℎ2)

(cid:17)

,

32

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

where we used symmetry ∫ 𝑡𝐾(𝑡)𝑑𝑡 = 0 and isotropy ∫ 𝑡𝑡⊤𝐾(𝑡)𝑑𝑡 = 𝜇2(𝐾)𝐼𝑑, which implies ∫ 𝑡⊤∇2𝑝(𝑥)𝑡𝐾(𝑡)𝑑𝑡 =
𝜇2(𝐾)tr(∇2𝑝(𝑥)) = 𝜇2(𝐾)Δ𝑝(𝑥). Similarly, for the numerator,

E[𝐵𝑛(𝑥)] = 𝑛E(cid:2)𝐾 ℎ(𝑥 − 𝑋)𝑌(cid:3) = 𝑛

∫

𝐾(𝑡)(𝑚𝑝)(𝑥 − ℎ𝑡)𝑑𝑡

∫

= 𝑛

(cid:16)

𝐾(𝑡)

(𝑚𝑝)(𝑥) − ℎ𝑡⊤∇(𝑚𝑝)(𝑥) +

𝑡⊤∇2(𝑚𝑝)(𝑥)𝑡 + 𝑜(ℎ2)

(cid:17)

𝑑𝑡

ℎ2
2

𝜇2(𝐾)tr(cid:0)∇2(𝑚𝑝)(𝑥)(cid:1) + 𝑜(ℎ2)

(cid:17)

(cid:16)

(cid:16)

= 𝑛

= 𝑛

𝑚(𝑥)𝑝(𝑥) +

𝑚(𝑥)𝑝(𝑥) +

ℎ2
2
ℎ2𝜇2(𝐾)
2

(cid:0)𝑝Δ𝑚 + 𝑚Δ𝑝 + 2∇𝑚⊤∇𝑝(cid:1)(𝑥) + 𝑜(ℎ2)

(cid:17)

,

where the last step uses the fact that tr(cid:0)∇2(𝑚𝑝)(cid:1) = 𝑝Δ𝑚 + 𝑚Δ𝑝 + 2∇𝑚⊤∇𝑝 by the product rule and symmetry of mixed
derivatives.

Now expand the ratio E[𝐵𝑛 (𝑥)]
E[𝐴𝑛 (𝑥)]

using the identity

𝑎0 + ℎ2𝑎2 + 𝑜(ℎ2)
𝑏0 + ℎ2𝑏2 + 𝑜(ℎ2)

=

𝑎0
𝑏0

+ ℎ2 𝑎2𝑏0 − 𝑎0𝑏2

𝑏2
0

+ 𝑜(ℎ2),

with 𝑎0 = 𝑚(𝑥)𝑝(𝑥), 𝑎2 =

𝜇2(𝐾)
2

(cid:0)𝑝Δ𝑚 + 𝑚Δ𝑝 + 2∇𝑚⊤∇𝑝(cid:1)(𝑥), 𝑏0 = 𝑝(𝑥), and 𝑏2 =

𝜇2(𝐾)

2 Δ𝑝(𝑥). This yields

E[𝐵𝑛(𝑥)]
E[𝐴𝑛(𝑥)]

= 𝑚(𝑥) +

= 𝑚(𝑥) +

ℎ2𝜇2(𝐾)
2
ℎ2𝜇2(𝐾)
2

(cid:0)𝑝Δ𝑚 + 𝑚Δ𝑝 + 2∇𝑚⊤∇𝑝(cid:1) 𝑝 − 𝑚𝑝Δ𝑝
𝑝2

(cid:12)
(cid:12)
(cid:12)𝑥

+ 𝑜(ℎ2)

(cid:16)

Δ𝑚(𝑥) + 2∇𝑚(𝑥)⊤ ∇𝑝(𝑥)
𝑝(𝑥)

(cid:17)

+ 𝑜(ℎ2),

which recovers our statement. Variance. Linearize
To leading order,

𝑚(𝑥) = 𝐵𝑛(𝑥)/𝐴𝑛(𝑥) around (E[𝐵𝑛(𝑥)], E[𝐴𝑛(𝑥)]) and use independence.
(cid:98)

Var[

𝑚(𝑥)] ≈
(cid:98)

Var[𝐵𝑛(𝑥)]
(E[𝐴𝑛(𝑥)])2

.

Compute

while

Therefore,

Var[𝐵𝑛(𝑥)] =

𝑛
(cid:213)

𝑖=1

Var(cid:0)𝐾 ℎ(𝑥 − 𝑋𝑖)𝑌𝑖 (cid:1)

(independence)

= 𝑛E(cid:2)𝐾 ℎ(𝑥 − 𝑋)2Var(𝑌 | 𝑋)(cid:3) = 𝑛E(cid:2)𝐾 ℎ(𝑥 − 𝑋)2𝑣(𝑋)(cid:3)

∫

= 𝑛

ℎ−2𝑑𝐾

(cid:17) 2

(cid:16) 𝑥 − 𝑢
ℎ

𝑣(𝑢)𝑝(𝑢)𝑑𝑢

∫

= 𝑛 ℎ−𝑑

𝐾(𝑡)2𝑣(𝑥 − ℎ𝑡)𝑝(𝑥 − ℎ𝑡)𝑑𝑡 = 𝑛 ℎ−𝑑 (cid:16)

𝑅(𝐾)𝑣(𝑥)𝑝(𝑥) + 𝑜(1)

(cid:17)

,

E[𝐴𝑛(𝑥)] = 𝑛 (cid:0)𝑝(𝑥) + 𝑜(1)(cid:1) .

Var[

𝑚(𝑥)] ≈
(cid:98)

𝑛 ℎ−𝑑𝑅(𝐾)𝑣(𝑥)𝑝(𝑥)
𝑛2𝑝(𝑥)2

=

𝑅(𝐾)
𝑛 ℎ𝑑

𝑣(𝑥)
𝑝(𝑥)

+ 𝑜 (cid:0)(𝑛 ℎ𝑑)−1(cid:1) ,

completing the proof.

□

B.6 Proof of Equation (5) to Equation (6)
Proof. Let ¯z = 1
𝑉𝑔

z𝑛,𝑣 denote the mean of the first 𝑉𝑔 vectors.

(cid:205)𝑉𝑔
𝑣=1

33

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

We prove that:

Expanding the left-hand side:

1
𝑉𝑔

𝑉𝑔
(cid:213)

𝑣=1

1
𝑉

𝑉
(cid:213)

𝑣′=1

∥z𝑛,𝑣 − z𝑛,𝑣′∥2

2 =

1
𝑉

𝑉
(cid:213)

𝑣′=1

∥¯z − z𝑛,𝑣′ ∥2
2

LHS =

=

=

=

1
𝑉𝑔𝑉

1
𝑉𝑔𝑉

𝑉𝑔
(cid:213)

𝑉
(cid:213)

𝑣=1
𝑉𝑔
(cid:213)

𝑣′=1

𝑉
(cid:213)

𝑣=1

𝑣′=1

∥z𝑛,𝑣 − z𝑛,𝑣′ ∥2
2

(cid:0)∥z𝑛,𝑣∥2

2 − 2z𝑇

𝑛,𝑣z𝑛,𝑣′ + ∥z𝑛,𝑣′ ∥2
2

(cid:1)

1
𝑉𝑔

1
𝑉𝑔

𝑉𝑔
(cid:213)

𝑣=1
𝑉𝑔
(cid:213)

𝑣=1

∥z𝑛,𝑣∥2

2 −

∥z𝑛,𝑣∥2

2 −

2
𝑉𝑔𝑉

2
𝑉

¯z𝑇

𝑉𝑔
(cid:213)

𝑉
(cid:213)

𝑣=1

𝑣′=1

z𝑇
𝑛,𝑣z𝑛,𝑣′ +

1
𝑉

𝑉
(cid:213)

𝑣′=1

∥z𝑛,𝑣′ ∥2
2

𝑉
(cid:213)

𝑣′=1

z𝑛,𝑣′ +

1
𝑉

𝑉
(cid:213)

𝑣′=1

∥z𝑛,𝑣′∥2
2

Expanding the right-hand side:

RHS =

1
𝑉

𝑉
(cid:213)

𝑣′=1

(cid:0)∥¯z∥2

2 − 2¯z𝑇z𝑛,𝑣′ + ∥z𝑛,𝑣′∥2

2

(cid:1)

= ∥¯z∥2

2 −

2
𝑉

¯z𝑇

𝑉
(cid:213)

𝑣′=1

z𝑛,𝑣′ +

1
𝑉

𝑉
(cid:213)

𝑣′=1

∥z𝑛,𝑣′∥2
2

To complete the proof, we verify that:

Expanding the right-hand side:

1
𝑉𝑔

𝑉𝑔
(cid:213)

𝑣=1

∥z𝑛,𝑣∥2

2 = ∥¯z∥2

2

∥¯z∥2

2 =

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

1
𝑉𝑔

𝑉𝑔
(cid:213)

𝑣=1

z𝑛,𝑣

2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
2

=

=

1
𝑉 2
𝑔

1
𝑉𝑔

𝑉𝑔
(cid:213)

𝑉𝑔
(cid:213)

z𝑇
𝑛,𝑣z𝑛,𝑣′′

𝑣′′=1

∥z𝑛,𝑣∥2
2

𝑣=1
𝑉𝑔
(cid:213)

𝑣=1

Therefore, LHS = RHS, completing the proof.

B.7 Proof of thm. 8
Proof. For each 𝑥,

Bias[

𝑚(𝑥)] =
(cid:98)

ℎ2𝜇2(𝐾)
2

(cid:16)

Δ𝑚(𝑥) + 2∇𝑚(𝑥)⊤∇ log 𝑝(𝑥)

(cid:17)

+ 𝑜(ℎ2).

34

(14)

(15)

(16)

(17)

(18)

(19)

(20)

(21)

(22)

(23)

(24)

□

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Square and integrate against 𝑝(𝑥):

ℬ2(ℎ; 𝑝, 𝑚) =

≤

=

(cid:16) ℎ2𝜇2(𝐾)
2
(cid:16) ℎ2𝜇2(𝐾)
2
(cid:16) ℎ2𝜇2(𝐾)
2

(cid:17) 2 ∫ (cid:16)

(cid:17) 2 ∫ (cid:16)

∫

(cid:17) 2 (cid:16)

2

Δ𝑚(𝑥) + 2∇𝑚(𝑥)⊤∇ log 𝑝(𝑥)

(cid:17) 2

𝑝(𝑥)𝑑𝑥 + 𝑜(ℎ4)

2(Δ𝑚(𝑥))2 + 2(2∇𝑚(𝑥)⊤∇ log 𝑝(𝑥))2(cid:17)

𝑝(𝑥)𝑑𝑥 + 𝑜(ℎ4)

(Δ𝑚(𝑥))2𝑝(𝑥)𝑑𝑥 + 8

∫

(∇𝑚(𝑥)⊤∇ log 𝑝(𝑥))2𝑝(𝑥)𝑑𝑥

(cid:17)

+ 𝑜(ℎ4),

where we used (𝑎 + 𝑏)2 ≤ 2𝑎2 + 2𝑏2 pointwise. Since |Δ𝑚(𝑥)| ≤ 𝐵 for all 𝑥, we have

∫

∫

(Δ𝑚)2𝑝 ≤

𝐵2𝑝 = 𝐵2.

For the second term, first use Cauchy–Schwarz and then integrate against 𝑝(𝑥) to obtain

(∇𝑚(𝑥)⊤∇ log 𝑝(𝑥))2 ≤ ∥∇𝑚(𝑥)∥2∥∇ log 𝑝(𝑥)∥2 ≤ 𝐿2∥∇ log 𝑝(𝑥)∥2
∫

(∇𝑚(𝑥)⊤∇ log 𝑝(𝑥))2𝑝(𝑥)𝑑𝑥 ≤ 𝐿2

∥∇ log 𝑝(𝑥)∥2𝑝(𝑥)𝑑𝑥 = 𝐿2𝐽(𝑝).

∫

=⇒

which can be combined with the bounds above to obtain the desired result. We similarly have for the integrated variance

𝒱 (ℎ; 𝑝) =

∫ (cid:16) 𝑅(𝐾)
𝑛 ℎ𝑑

𝑣(𝑥)
𝑝(𝑥)

+ 𝑜 (cid:0)(𝑛 ℎ𝑑)−1(cid:1) (cid:17)

𝑝(𝑥)𝑑𝑥 =

∫

𝑅(𝐾)
𝑛 ℎ𝑑

𝑣(𝑥)𝑑𝑥 + 𝑜 (cid:0)(𝑛 ℎ𝑑)−1(cid:1) ,

which is independent of 𝑝.

□

B.8 Proof of lemma. 3
Proof. We first start by reminding the reader about the original Cramér-Wold theorem that is a function of all possible
directions (not unit-norm ones).

Theorem 10: Cramér-Wold Cramér and Wold [1936]

Let 𝑋 and 𝑌 be random vectors in R𝐷:

𝑋

𝑑
= 𝑌 ⇐⇒ ⟨𝑋 , 𝑎⟩

𝑑
= ⟨𝑌, 𝑎⟩, ∀𝒂 ∈ R𝐷 .

(25)

Our proof will follow the same proof as for thm. 10. Necessity is immediate: if 𝑋 𝑑

= 𝑌, then every measurable function
of 𝑋 has the same distribution as the corresponding function of 𝑌, from which the linear mapping 𝑥 ↦→ ⟨𝑢, 𝑥⟩ for 𝑢 ∈ S𝑑−1
is a special case. For sufficiency, assume ⟨𝑢, 𝑋⟩ 𝑑
= ⟨𝑢, 𝑌⟩ for all 𝑢 ∈ S𝑑−1. Let 𝜑𝑋 (𝑡) := E(cid:2)𝑒 𝑖⟨𝑡,𝑋⟩(cid:3) and 𝜑𝑌(𝑡) := E(cid:2)𝑒 𝑖⟨𝑡,𝑌⟩(cid:3)
denote the characteristic functions of 𝑋 and 𝑌. Fix an arbitrary 𝑡 ∈ R𝑑; if 𝑡 = 0, then 𝜑𝑋 (0) = 𝜑𝑌(0) = 1. If 𝑡 ≠ 0, write
𝑡 = 𝑠𝑢 with 𝑠 := ∥𝑡∥ > 0 and 𝑢 := 𝑡/∥𝑡∥ ∈ S𝑑−1. By the assumption, ⟨𝑢, 𝑋⟩ 𝑑

= ⟨𝑢, 𝑌⟩, hence for this 𝑢 and 𝑠 we have

𝜑𝑋 (𝑡) = E(cid:2)𝑒 𝑖⟨𝑡,𝑋⟩(cid:3) = E(cid:2)𝑒 𝑖𝑠⟨𝑢,𝑋⟩(cid:3) = E(cid:2)𝑒 𝑖𝑠⟨𝑢,𝑌⟩(cid:3) = E(cid:2)𝑒 𝑖⟨𝑡,𝑌⟩(cid:3) = 𝜑𝑌(𝑡).

Thus 𝜑𝑋 (𝑡) = 𝜑𝑌(𝑡) for all 𝑡 ∈ R𝑑, i.e., 𝜑𝑋 ≡ 𝜑𝑌 on R𝑑. By the uniqueness theorem for characteristic functions, this
implies 𝑋 𝑑
= 𝑌. (ii) Define 𝜓𝑛,𝑡 := E(cid:2)𝑒 𝑖⟨𝑡,𝑋𝑛 ⟩(cid:3) and 𝜓𝑡 := E(cid:2)𝑒 𝑖⟨𝑡,𝑋⟩(cid:3). Fix 𝑡 ∈ R𝑑 and decompose 𝑡 = 𝑠𝑢 with 𝑠 := ∥𝑡∥ ≥ 0
and 𝑢 ∈ S𝑑−1 (take, e.g., 𝑢 = 𝑡/∥𝑡∥ if 𝑡 ≠ 0, and any 𝑢 if 𝑡 = 0). The map 𝑔𝑠 : R → R, 𝑔𝑠(𝑥) = 𝑠𝑥, is continuous. By the
continuous mapping theorem applied to the real-valued random variables ⟨𝑢, 𝑋𝑛⟩

𝑑
−→ ⟨𝑢, 𝑋⟩, we obtain

⟨𝑡, 𝑋𝑛⟩ = 𝑠⟨𝑢, 𝑋𝑛⟩

𝑑
−→ 𝑠⟨𝑢, 𝑋⟩ = ⟨𝑡, 𝑋⟩.

35

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Hence, for every fixed 𝑡 ∈ R𝑑, the one-dimensional projections satisfy ⟨𝑡, 𝑋𝑛⟩
convergence of characteristic functions:

𝑑
−→ ⟨𝑡, 𝑋⟩, which in turn yields pointwise

𝜓𝑛,𝑡 = E(cid:2)𝑒 𝑖⟨𝑡,𝑋𝑛 ⟩(cid:3) −→ E(cid:2)𝑒 𝑖⟨𝑡,𝑋⟩(cid:3) = 𝜓𝑡 ,

for all 𝑡 ∈ R𝑑.

Therefore, by Lévy’s continuity theorem, 𝑋𝑛

𝑑
−→ 𝑋. This completes the proof.

□

B.9 Proof of thm. 2
Proof. We first formulate the following assumptions required for the proof–all of this are satisfied by typical univariate
statistical tests.

𝑃 = 𝑄 if and only if 𝑃𝑎 = 𝑄𝑎 for all 𝑎 ∈ 𝑆𝑑−1 (population-level equivalence of laws).
𝐴𝑛 are finite sets with mesh Δ(𝐴𝑛) := sup𝑢∈𝑆𝑑−1 min𝑎∈𝐴𝑛 ∥𝑢 − 𝑎∥ → 0 as 𝑛 → ∞.
If 𝑃 ≠ 𝑄, there exists a separating direction 𝑎★ ∈ 𝑆𝑑−1 and a neighborhood 𝑈 of 𝑎★ such that

inf
𝑎∈𝑈

lim
𝑛→∞

Pr (cid:0)𝑇𝑎,𝑛 ≥ 𝑢𝑛(𝛼)(cid:1) = 1.

(Intuitively: near a truly separating direction, the 1D statistic eventually exceeds the global null threshold with probability
→ 1.)

(i) Under 𝐻0 : 𝑃 = 𝑄, our assumption implies no separating direction exists at the population level, and the
calibration of 𝑢𝑛(𝛼) ensures Pr(𝑀𝑛 ≥ 𝑢𝑛(𝛼)) ≤ 𝛼 for all 𝑛, hence lim sup𝑛→∞ Pr(Ψ𝑛 = 1) ≤ 𝛼. (ii) Suppose 𝑃 ≠ 𝑄. Our
assumption guarantees that there exists at least one separating direction 𝑎★ with 𝑃𝑎★ ≠ 𝑄𝑎★. Our assumption guarantees
a neighborhood 𝑈 of 𝑎★ in which the projection statistics exceed the global null threshold with probability tending to 1:

inf
𝑎∈𝑈

lim
𝑛→∞

Pr (cid:0)𝑇𝑎,𝑛 ≥ 𝑢𝑛(𝛼)(cid:1) = 1.

By assumption, for all large 𝑛 the set 𝐴𝑛 contains at least one direction 𝑎𝑛 ∈ 𝑈 (dense coverage). Therefore,

Pr(Ψ𝑛 = 1) = Pr (cid:0)𝑀𝑛 ≥ 𝑢𝑛(𝛼)(cid:1) ≥ Pr (cid:0)𝑇𝑎𝑛 ,𝑛 ≥ 𝑢𝑛(𝛼)(cid:1) −→ 1,

which proves consistency.

□

B.10 Proof of thm. 5
Proof. For each case, consider the function 𝑔(𝑎) on S𝐷−1 defined by the quantity of interest (CF, CDF, or moment) at a
fixed 𝑡 or 𝑘. Since 𝑓 ∈ 𝐻 𝛼(R𝐷), the mapping 𝑎 ↦→ 𝑔(𝑎) is in 𝐻 𝛼(S𝐷−1) for each fixed 𝑡 or 𝑘.

Given 𝑀 samples {𝑎𝑖}𝑀
𝑖=1

on the sphere, the best possible reconstruction of 𝑔 from its values at these points is given by
spherical interpolation. By classical results on Sobolev spaces and spherical harmonics (see, e.g., Narcowich et al. [2006]),
the 𝐿2 interpolation error for functions in 𝐻 𝛼(S𝐷−1) using 𝑀 points is bounded by

E𝑏

(cid:2)|𝑔(𝑏) − 𝑔∗(𝑏)|2(cid:3) ≤ 𝐶(𝐷, 𝛼)𝑀−2𝛼/(𝐷−1)∥𝑔∥2

𝐻 𝛼(S𝐷−1)

,

where 𝑔∗ is the interpolant matching 𝑔 at the 𝑀 sampled points. The interpolation error bound on the sphere follows
from the theory of spherical harmonics and Marcinkiewicz–Zygmund (MZ) inequalities . Any 𝑓 ∈ 𝐻 𝛼(S𝑑) admits a
spherical harmonics expansion, and the best 𝐿2 approximation by harmonics of degree at most 𝐿 satisfies

∥ 𝑓 − 𝑃𝐿 𝑓 ∥𝐿2(S𝑑) ≤ (1 + 𝐿2)−𝛼/2∥ 𝑓 ∥𝐻 𝛼(S𝑑),

where 𝑃𝐿 𝑓 is the projection onto harmonics of degree ≤ 𝐿 [Narcowich et al., 2006, Lemma 2.1]. If 𝑀 points are distributed
quasi-uniformly on S𝑑, then for 𝐿 ∼ 𝑐𝑀1/𝑑, the set forms a Marcinkiewicz–Zygmund (MZ) set for degree 𝐿 [Mhaskar
et al., 2001, Theorem 1.1]. This allows reconstruction of any function in the space of harmonics of degree at most 𝐿 from
its values at these points, and the 𝐿2 interpolation error for 𝑓 is bounded by

∥ 𝑓 − 𝐼𝑀 𝑓 ∥𝐿2(S𝑑) ≤ 𝐶(1 + 𝐿2)−𝛼/2∥ 𝑓 ∥𝐻 𝛼(S𝑑),

36

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

where 𝐼𝑀 𝑓 is any interpolant matching 𝑓 at the 𝑀 points [Narcowich et al., 2006, Theorem 3.1]. Substituting 𝐿 ∼ 𝑐𝑀1/𝑑
yields the rate 𝑀−𝛼/𝑑, and thus

E𝜔| 𝑓 (𝜔) − 𝐼𝑀 𝑓 (𝜔)|2 ≤ 𝐶(𝑑, 𝛼)𝑀−2𝛼/𝑑∥ 𝑓 ∥2

𝐻 𝛼(S𝑑)

,

with explicit 𝐶(𝑑, 𝛼) as in the main theorem. Integrating (or summing) over 𝑡 (for CF and CDF) or 𝑘 (for moments, with
weights 𝑤𝑘) yields the stated bounds. The explicit constant 𝐶(𝐷, 𝛼) arises from the theory of spherical Sobolev spaces
and is given above.

For the moment case, the sum over 𝑘 is weighted to ensure convergence, as higher moments may grow rapidly. The

weights 𝑤𝑘 can be chosen, for example, as 𝑤𝑘 = 1/𝑘!.

This completes the proof.

□

B.11 Proof of thm. 3
Pick distinct 𝑥0, . . . , 𝑥𝐾+1 ∈ R and consider the linear map 𝐴 : R𝐾+2 → R𝐾+1, (𝐴𝑝)𝑟 = (cid:205)𝐾+1
for 𝑟 = 0, . . . , 𝐾. Then
rank(𝐴) ≤ 𝐾 + 1, so ker(𝐴) ≠ {0}. Let 𝑣 ∈ ker(𝐴) \ {0}; from (𝐴𝑝)0 = (cid:205)𝑗 𝑝 𝑗, we get (cid:205)𝑗 𝑣 𝑗 = 0, hence 𝑣 has positive and
negative entries. Choose a strictly positive probability vector 𝑝 and 𝜀 > 0 small such that 𝑝± := 𝑝 ± 𝜀𝑣 remain probability
vectors. Then 𝐴𝑝+ = 𝐴𝑝−, so the distributions supported on {𝑥 𝑗} with masses 𝑝± are distinct yet match moments up to
order 𝐾.

𝑗=0 𝑝 𝑗 𝑥𝑟

𝑗

B.12 Proof of thm. 4
Proof. Fix the Gaussian weight

and define the population CF distance

Let the empirical CF be

and consider the V-statistic estimator

𝑤𝑠(𝑡) = 𝑒−𝑠2𝑡2 ,

𝑠 > 0,

𝐷(𝑃, 𝐺) =

∫

R

𝑤𝑠(𝑡)(cid:12)

2𝑑𝑡.
(cid:12)𝜑𝑃(𝑡) − 𝜑𝐺(𝑡)(cid:12)
(cid:12)

𝜑𝑁 (𝑡) =
(cid:98)

1
𝑁

𝑁
(cid:213)

𝑖=1

𝑒 𝑖𝑡𝑋𝑖 ,

(cid:98)𝐷𝑉 =

∫

R

𝑤𝑠(𝑡)(cid:12)

2𝑑𝑡.
𝜑𝑁 (𝑡) − 𝜑𝐺(𝑡)(cid:12)
(cid:12)
(cid:12)(cid:98)

We use only that |𝑒 𝑖𝑡𝑋 | = 1, |𝜑𝑃(𝑡)| ≤ 1, |𝜑𝐺(𝑡)| ≤ 1, and integrability of 𝑤𝑠. For each 𝑖 differentiate under the integral
(dominated convergence applies because the integrand and its derivative are bounded)

since |

𝜑𝑁 (𝑡)| ≤ 1 and |𝜑𝐺(𝑡)| ≤ 1,
(cid:98)

𝜕 (cid:98)𝐷𝑉
𝜕𝑋𝑖
𝜑𝑁 (𝑡)
(cid:98)
𝜕𝑋𝑖

𝜕

=

=

∫

(cid:16)(cid:0)
𝑤𝑠(𝑡)2ℜ

R
1
𝑁

𝑖𝑡𝑒 𝑖𝑡𝑋𝑖 ,

𝜑𝑁 (𝑡) − 𝜑𝐺(𝑡)(cid:1)
(cid:98)

𝜕

(cid:17)

𝜑𝑁 (𝑡)
(cid:98)
𝜕𝑋𝑖

𝑑𝑡,

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

𝜕 (cid:98)𝐷𝑉
𝜕𝑋𝑖

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤

≤

=

∫

2
𝑁

∫

4
𝑁
4
𝑁 𝑠2

,

𝑤𝑠(𝑡)|𝑡|(cid:0)|

𝜑𝑁 (𝑡)| + |𝜑𝐺(𝑡)|(cid:1) 𝑑𝑡
(cid:98)

𝑤𝑠(𝑡)|𝑡|𝑑𝑡

37

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

using ∫
R

𝑒−𝑠2𝑡2|𝑡|𝑑𝑡 = 1/𝑠2.

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

𝜕 (cid:98)𝐷𝑉
𝜕𝑋𝑖

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤

4
𝑁

∫

R

𝑤𝑠(𝑡)|𝑡|𝑑𝑡 =

4
𝑁 𝑠2

.

Moreover, differentiating once more in 𝑋𝑖 and using |

𝜑𝑁 (𝑡)| ≤ 1, |𝜑𝐺(𝑡)| ≤ 1 gives a global Lipschitz bound
(cid:98)

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

𝜕2
(cid:98)𝐷𝑉
𝜕𝑋2
𝑖

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤

𝐶
𝑁

∫

R

𝑤𝑠(𝑡)𝑡2𝑑𝑡 =

√

𝜋
2𝑠3

,

𝐶
𝑁

·

for some absolute constant 𝐶 arising from bounded factors and product rule. Hence ECF gradients are uniformly
bounded and Lipschitz, with scale controlled only by (𝑁 , 𝑠).

(B) (Moment sample-gradients are polynomial in 𝑋𝑖 and unbounded for 𝑘 ≥ 2.) Let (cid:98)𝐷𝑉 be as above. Define the moment
objective

(cid:98)𝐷𝑘 = ( ¯𝜙 − 𝜇)⊤𝑊( ¯𝜙 − 𝜇),

¯𝜙 :=

1
𝑁

𝑁
(cid:213)

𝑖=1

𝜙(𝑋𝑖), 𝜙(𝑥) = (𝑥, 𝑥2, . . . , 𝑥 𝑘)⊤,

for a symmetric positive semidefinite 𝑊 ∈ R𝑘×𝑘 and Gaussian target moments 𝜇 = E𝐺[𝜙(𝑌)]. For each 𝑖,

𝜕 (cid:98)𝐷𝑘
𝜕𝑋𝑖
𝜕𝜙(𝑋)
𝜕𝑋

2
𝑁

=

( ¯𝜙 − 𝜇)⊤𝑊

𝜕𝜙(𝑋𝑖)
𝜕𝑋𝑖

,

=(cid:0)1, 2𝑋 , 3𝑋2, . . . , 𝑘𝑋 𝑘−1(cid:1) ⊤.

The gradient formula follows by the chain rule and linearity of ¯𝜙. Let 𝑐 := 𝑊( ¯𝜙 − 𝜇) and write 𝑐𝑟 for its 𝑟-th coordinate.
Then

𝑟=1
which is a polynomial in 𝑋𝑖 of degree deg = max{𝑟 − 1 : 𝑐𝑟 ≠ 0} ≤ 𝑘 − 1. In particular, if 𝑐𝑘 ≠ 0 (the generic case when
the top-weighted deviation is nonzero), then

𝜕 (cid:98)𝐷𝑘
𝜕𝑋𝑖

=

2
𝑁

𝑘
(cid:213)

𝑐𝑟 𝑟𝑋 𝑟−1
𝑖

,

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

𝜕 (cid:98)𝐷𝑘
𝜕𝑋𝑖

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

−−−−−→
|𝑋𝑖 |→∞

∞ as

|𝑋𝑖|𝑘−1.

The expression is a nonconstant polynomial in 𝑋𝑖 of degree deg ≤ 𝑘 − 1 whenever some 𝑐𝑟 ≠ 0 with 𝑟 ≥ 2. Thus the
gradient cannot be uniformly bounded on R. If 𝑐𝑘 ≠ 0, the leading term dominates and the magnitude grows like |𝑋𝑖|𝑘−1,
□
proving unboundedness for 𝑘 ≥ 2.

B.13 Proof of thm. 6
Proof. A direct calculation shows Fix 𝑡 ∈ R𝑑 and abbreviate 𝑍𝑗 (cid:66) 𝑒i𝑡⊤𝑋𝑗 , so that 𝜙𝑛(𝑡) = 1
(cid:205)𝑛
almost surely (since 𝑡⊤𝑋𝑗 ∈ R), and E[𝑍𝑗] = 𝜙𝜃(𝑡) for all 𝑗. We start from the algebraic identity

𝑛

𝑗=1 𝑍𝑗. Note that |𝑍𝑗| = 1

2
(cid:12)𝜙𝑛(𝑡) − 𝜓(𝑡)(cid:12)
(cid:12)
(cid:12)

= 𝜙𝑛(𝑡)𝜙𝑛(𝑡) − 𝜓(𝑡)𝜙𝑛(𝑡) − 𝜓(𝑡)𝜙𝑛(𝑡) + (cid:12)

2.
(cid:12)𝜓(𝑡)(cid:12)
(cid:12)

38

(26)

(27)

(28)

(29)

(30)

(31)

(32)

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Taking expectations term by term gives

E (cid:104)(cid:12)

2(cid:105)
(cid:12)𝜙𝑛 − 𝜓(cid:12)
(cid:12)

=E (cid:2)|𝜙𝑛|2(cid:3) − 𝜓E (cid:104)

𝜙𝑛

(cid:105)

− 𝜓E (cid:2)𝜙𝑛

(cid:3) + |𝜓|2,

𝑛
(cid:213)

E[𝑍𝑗] + |𝜓|2,

1
𝑛

=E (cid:2)|𝜙𝑛|2(cid:3) − 𝜓E[𝜙𝑛] − 𝜓

𝑗=1
=E (cid:2)|𝜙𝑛|2(cid:3) − 𝜓𝜙𝜃 − 𝜓𝜙𝜃 + |𝜓|2,
=E (cid:2)|𝜙𝑛|2(cid:3) − 2Re(cid:0)𝜓𝜙𝜃(cid:1) + |𝜓|2,
2
(cid:12)
(cid:12)

(cid:12)

(cid:12)

(cid:12)

(cid:12)


E (cid:104)
𝑍𝑗 𝑍𝑙

(cid:12)

(cid:12)

(cid:12)

(cid:12)

(cid:12)

(cid:12)



𝑛
(cid:213)

𝑛
(cid:213)

𝑛
(cid:213)

=E

1
𝑛

𝑍𝑗

𝑗=1

=

(cid:105)

1
𝑛2

𝑗=1

𝑙=1

− 2Re(cid:0)𝜓𝜙𝜃(cid:1) + |𝜓|2,

− 2Re(cid:0)𝜓𝜙𝜃(cid:1) + |𝜓|2,

Since the 𝑍𝑗 are i.i.d.,

hence

E (cid:104)

𝑍𝑗 𝑍𝑙

(cid:105)

=

(cid:40)E (cid:2)|𝑍1|2(cid:3) = 1,
E[𝑍𝑗]E[𝑍𝑙] = 𝜙𝜃 𝜙𝜃 = |𝜙𝜃|2,

if 𝑗 = 𝑙,

if 𝑗 ≠ 𝑙,

E (cid:2)|𝜙𝑛|2(cid:3) =

=

1
𝑛2
1
𝑛

(cid:16)

𝑛 + 𝑛(𝑛 − 1)|𝜙𝜃|2(cid:17)
(cid:19)
(cid:18)

+

1 −

|𝜙𝜃|2

1
𝑛

Plugging these, we obtain

=|𝜙𝜃|2 +

1 − |𝜙𝜃|2
𝑛

E (cid:104)(cid:12)

2(cid:105)
(cid:12)𝜙𝑛 − 𝜓(cid:12)
(cid:12)

=

(cid:18)

|𝜙𝜃|2 +

(cid:19)

1 − |𝜙𝜃|2
𝑛

− 2Re(cid:0)𝜓𝜙𝜃(cid:1) + |𝜓|2

= (cid:0)|𝜙𝜃|2 − 2Re(cid:0)𝜓𝜙𝜃(cid:1) + |𝜓|2(cid:1) +

1 − |𝜙𝜃|2
𝑛

= (cid:12)

2 +
(cid:12)𝜙𝜃 − 𝜓(cid:12)
(cid:12)

1 − |𝜙𝜃|2
𝑛

.

Under Dominated convergence, E[∇𝜃𝐷𝑛(𝑡)] = ∇𝜃E[𝐷𝑛(𝑡)], hence

E [∇𝜃𝐷𝑛(𝑡)] = ∇𝜃

2 + ∇𝜃
(cid:12)𝜙𝜃(𝑡) − 𝜓(𝑡)(cid:12)
(cid:12)
(cid:12)

1 − |𝜙𝜃(𝑡)|2
𝑛

,

concluding the proof.

In practice one replaces ∫
R

𝑤(𝑡)(·)𝑑𝑡 by a deterministic quadrature on a uniform grid 𝑡𝑘 ∈ [−𝑇, 𝑇] with weights 𝜔𝑘 (e.g.
trapezoidal rule) and a Gaussian window 𝑤(𝑡) = 𝑒−𝛼𝑡2. All statements above remain valid with the integral replaced by
(cid:205)𝑘 𝜔𝑘(·):

𝐿(𝜃) ≈ (cid:213)

𝜔𝑘

2, (cid:98)𝐿𝑛(𝜃) ≈ (cid:213)
(cid:12)𝜙𝜃(𝑡𝑘) − 𝜓(𝑡𝑘)(cid:12)
(cid:12)
(cid:12)

𝜔𝑘

2,
(cid:12)𝜙𝑛(𝑡𝑘) − 𝜓(𝑡𝑘)(cid:12)
(cid:12)
(cid:12)

𝑘

𝑘

39

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

and the bias term becomes

Bias(𝜃) = −

1
𝑛

(cid:213)

𝑘

𝜔𝑘∇𝜃

2.
(cid:12)
(cid:12)𝜙𝜃(𝑡𝑘)(cid:12)
(cid:12)

Since the grid and weights are deterministic, they do not affect unbiasedness with respect to sampling; they only
introduce a deterministic approximation error to the target functional 𝐿(𝜃).

B.14 Proof of VICReg’s Recovery
Proof. We prove this result in two parts.

Part I: E[X] = 0 Given that E[⟨X, a⟩] = 0 for all unit vectors a, and noting that ⟨X, a⟩ = a𝑇X, we have:

By linearity of expectation:

E[a𝑇X] = 0

for all a ∈ R𝑑 with ∥a∥ = 1

a𝑇E[X] = 0

for all unit vectors a

□

(33)

(34)

Let 𝝁 = E[X]. We claim that 𝝁 = 0. Suppose, for the sake of contradiction, that 𝝁 ≠ 0. Then ∥𝝁∥2 > 0. Define the unit
vector:

a∗ =

𝝁
∥𝝁∥2

Since a∗ is a unit vector, equation (33) implies:

(a∗)𝑇𝝁 = 0

However, substituting the definition of a∗:

(a∗)𝑇𝝁 =

(cid:19)𝑇

(cid:18) 𝝁
∥𝝁∥2

𝝁 =

𝝁𝑇𝝁
∥𝝁∥2

∥𝝁∥2
2
∥𝝁∥2

=

= ∥𝝁∥2 > 0

This contradiction establishes that 𝝁 = 0.

Part II: Cov(X) = I𝑑

Since E[X] = 0, we have:

Expanding the quadratic form:

Var(⟨X, a⟩) = E[(⟨X, a⟩)2] = E[(a𝑇X)2]

E[(a𝑇X)2] = E[a𝑇XX𝑇a] = a𝑇E[XX𝑇]a

Since E[X] = 0, the covariance matrix is Cov(X) = E[XX𝑇]. Let 𝚺 = Cov(X). The variance condition gives us:

a𝑇𝚺a = 1

for all unit vectors a

We now show that 𝚺 = I𝑑. Step 1: Diagonal entries. For 𝑖 ∈ {1, 2, . . . , 𝑑}, let e𝑖 denote the 𝑖-th standard basis vector. Setting
a = e𝑖 in equation (40):

e𝑇
𝑖 𝚺e𝑖 = Σ𝑖𝑖 = 1
Therefore, all diagonal entries of 𝚺 equal 1. Step 2: Off-diagonal entries. For distinct indices 𝑖, 𝑗 ∈ {1, 2, . . . , 𝑑}, consider the
unit vector:

(41)

Applying equation (40):

a =

e𝑖 + e𝑗
∥e𝑖 + e𝑗∥2

e𝑖 + e𝑗
√

=

2

(35)

(36)

(37)

(38)

(39)

(40)

(42)

(43)

a𝑇𝚺a =

1
2

(e𝑖 + e𝑗)𝑇𝚺(e𝑖 + e𝑗) = 1

40

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Expanding the quadratic form and using the symmetry of 𝚺:

1
2

(e𝑇

𝑖 𝚺e𝑖 + 2e𝑇
1
2

𝑖 𝚺e𝑗 + e𝑇

𝑗 𝚺e𝑗) = 1

(Σ𝑖𝑖 + 2Σ𝑖𝑗 + Σ𝑗 𝑗) = 1

1
2

(1 + 2Σ𝑖𝑗 + 1) = 1

1 + Σ𝑖𝑗 = 1
Σ𝑖𝑗 = 0

Therefore, all off-diagonal entries of 𝚺 equal zero, establishing that 𝚺 = I𝑑.

C Background
Foundation: The Linear Regression Model We start with the standard linear regression model:

y = X𝜷 + 𝜺

where:

• y = [𝑦1, 𝑦2, . . . , 𝑦𝑛]𝑇 ∈ R𝑛 is the response vector

• X ∈ R𝑛×𝑝 is the design matrix with X𝑖𝑗 = 𝑥𝑖𝑗

• 𝜷 = [𝛽1, 𝛽2, . . . , 𝛽𝑝]𝑇 ∈ R𝑝 is the parameter vector

• 𝜺 = [𝜀1, 𝜀2, . . . , 𝜀𝑛]𝑇 ∼ 𝒩 (0, 𝜎2I𝑛) is the error vector

The error assumption means:

Step 1: Deriving the OLS Estimator To find the OLS estimator, we minimize the sum of squared residuals:

E[𝜀𝑖] = 0, Var(𝜀𝑖) = 𝜎2, Cov(𝜀𝑖 , 𝜀𝑗) = 0 for 𝑖 ≠ 𝑗

(44)

(45)

(46)

(47)

(48)

□

SSR(𝜷) =

𝑛
(cid:213)

(𝑦𝑖 − x𝑇

𝑖 𝜷)2 = (y − X𝜷)𝑇(y − X𝜷)

Expanding this quadratic form:

𝑖=1

SSR(𝜷) = y𝑇y − 2𝜷𝑇X𝑇y + 𝜷𝑇X𝑇X𝜷

(49)

Taking the derivative with respect to 𝜷:

Setting equal to zero and solving:

Assuming X𝑇X is invertible:

𝜕SSR
𝜕𝜷

= −2X𝑇y + 2X𝑇X𝜷

−2X𝑇y + 2X𝑇X𝜷 = 0
X𝑇X𝜷 = X𝑇y

ˆ𝜷 = (X𝑇X)−1X𝑇y

D Details on Low-Discrepancy Sequences
Quasi-Monte Carlo (QMC) methods, such as the Sobol sequence, are widely used to generate low-discrepancy samples
in the unit hypercube, providing improved uniformity over purely random sampling. To obtain samples uniformly
distributed on the hypersphere, each QMC point is mapped to a standard normal vector via the inverse cumulative

41

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Figure 15. Depiction of the expected BCS loss upper bound (thm. 5) for various smoothness values 𝛼. We clearly see that as the
smoothness increases (blue to red), as the upper bound decreases more and more rapidly with 𝑀.

Table 3. Performance metrics across different sample sizes from Figure 12

Freeze Backbone Model Name

Samples per Class

No

Yes

LeJEPA (Ours)
ConvNeXt-V2 Nano
LeViT-128
ResNet-18
ResNet-34

Baselines
DINOv2 Small
DINOv3 ViT-S/16

LeJEPA (Ours)
ConvNeXt-V2 Nano
LeViT-128
ResNet-18
ResNet-34

Baselines
DINOv2 Small
DINOv3 ViT-S/16

All

1

2

5

10

100

1000

82.72
79.41
82.15
83.28

29.42
18.45
23.34
24.27

36.65
24.08
31.56
31.51

50.94
33.11
43.82
44.23

59.85
41.76
54.64
53.95

75.34
64.59
73.53
74.93

81.97
77.59
81.41
82.32

78.34
81.60

21.05
24.71

21.71
29.43

30.33
37.71

36.23
44.71

60.81
69.87

75.55
80.54

76.52
69.00
75.95
78.17

28.74
25.85
30.48
31.08

36.65
33.30
38.22
38.33

50.60
45.52
50.85
52.26

59.50
52.43
58.86
60.63

72.62
64.37
72.70
74.77

77.24
69.39
76.39
78.62

67.62
71.38

27.68
30.17

32.22
36.65

40.72
45.74

47.72
51.51

62.49
65.90

67.89
71.35

Table 4. Top 1 accuracy (in %) with LeJEPA pretraining on Imagenet-100 for 400 epochs (All values are percentages)

backbone
Projector
w/ predictor w/ SWA

1-layer

resnet50
2-layer

3-layer

vit_small_patch8_224
2-layer

3-layer

1-layer

vit_tiny_patch8_224
2-layer

1-layer

3-layer

False

True

False
True
False
True

79.71
79.79
79.41
78.87

82.44
82.69
82.44
82.04

83.93
83.50
83.57
82.82

76.59
79.96
77.58
77.11

80.77
83.63
79.41
81.77

81.07
84.12
81.91
82.58

71.79
75.86
67.74
69.53

76.87
82.36
77.64
78.27

80.37
80.50
80.73
79.77

distribution function (CDF), and then projected onto the sphere by normalization. This approach leverages the rotational
invariance of the multivariate normal distribution, ensuring that the resulting directions are uniformly distributed on

42

255075100125150175NumberofdirectionsM−100−50050100150200250C(d,α)·M−2α/d(logscale)DecayofErrorBoundConstantvs.NumberofDirections(D=5)α=1α=21α=41α=61α=81α=101α=121α=141α=161α=181LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Table 5. Small architecture in-domain LeJEPA pretraining from random initialization across datasets and architectures, with frozen backbone
linear evaluation. First, LeJEPA is able to produce near state-of-the-art performances on tiny dataset with only a thousand samples, e.g., flowers102.
Second, on non-natural image data, LeJEPA clearly outperforms the latest frontier vision models, e.g., Galaxy10. See Figure 12 for additional
experiments with varying number of training samples and with full finetuning.

Pretraining
# train. samples

flowers102
1020

cifar100
50000

food101
75750

inet10
13000

cifar10
50000

galaxy10
11008

LeJEPA (convnextv2_nano) 14M in-domain
in-domain
LeJEPA (resnet18) 11M
in-domain
LeJEPA (resnet34) 21M
in-domain
LeJEPA (resnext26ts) 8M
LeJEPA (swin_tiny) 27M
in-domain
ĲEPA-inet22k (ViT-H/14) 630M inet1k

64.34
74.57
71.85
82.19
63.94
85.76

69.26
69.94
70.44
69.10
65.08
86.93

69.59
73.57
74.95
76.77
78.40
81.06

90.81
92.36
92.80
92.82
92.87
98.65

92.22
92.51
93.16
91.59
92.67
97.77

76.05
75.32
77.29
73.78
74.89
62.93

Table 6. Time (in millisecond) to compute the proposed SIGReg loss from algorithm 1 on a Tesla V100-SXM2-16GB for varying mini-batch size (𝑁),
number of slices (𝑀), integration points. Results are computed over 10 runs.

N

M

# integration
points

mean (ms)

std (ms)

512
512
512
2048
8192
8192
32768

512
512
512
512
512
8192
512
512 2048
512 8192

16
64
256
16
16
16
16
16
16

0.465236
0.461317
0.627644
1.406441
6.188304
8.685009
26.373118
0.465614
0.670379

0.011642
0.003894
0.003337
0.002415
0.007226
0.038829
0.012732
0.005274
0.006854

Table 7. Number of Figure 8.

resnet50

𝜆
#views

2
4
8

0.001

0.005

0.010

0.020

0.025

0.050

0.100

0.150

0.200

0.300

0.400

0.500

81.41
79.88
76.67

82.73
83.04
81.58

83.49
84.36
83.59

82.99
84.68
83.49

82.23
84.33
83.76

-
83.00
84.32

-
82.91
83.66

-
81.05
83.07

-
78.58
82.16

-
-
81.00

-
-
79.25

-
-
77.72

the sphere’s surface. While the low-discrepancy property is not strictly preserved under this nonlinear mapping, the
resulting samples are empirically more uniform than random samples and are standard in high-dimensional applications
Marsaglia [1972], Dick and Pillichshammer [2010], Caflisch [1998].
Require: Number of points 𝑁, dimension 𝑑
Ensure: Points {y𝑖}𝑁
𝑖=1
1: for 𝑖 = 1 to 𝑁 do
2:

quasi-uniformly distributed on S𝑑−1

⊲ Φ−1 is the inverse CDF of the standard normal

Generate x𝑖 ∈ [0, 1]𝑑 as the 𝑖-th point of a Sobol sequence
Transform each component: 𝑧𝑖,𝑗 = Φ−1(𝑥𝑖,𝑗) for 𝑗 = 1, . . . , 𝑑
Normalize: y𝑖 = z𝑖/∥z𝑖∥2

3:

4:
5: end for

E Shapiro-Wilk Test
Let X1 < X2 < . . . < Xn denote an ordered random sample of size n from a standard normal distribution. Also, let mÂ 5
(m1,m2,...,mn) be the vector of expected values of standard normal order statistics, and let V 5 (vĳ ) be the corresponding

43

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

n 3 n covariance matrix, so that

𝐸 (𝑋𝑖) = 𝑚𝑖

and

cov (cid:0)𝑋𝑖 , 𝑋𝑗 (cid:1) = 𝑣𝑖𝑗 ,

𝑖, 𝑗 = 1, 2, . . . , 𝑛

The W test statistic Shapiro and Wilk [1965] for normality is then denoted by

(aY)
𝑆2

((cid:205)𝑛
𝑖=1 𝑎𝑖𝑌𝑖)
𝑖=1(𝑌𝑖 − ¯𝑌)2 =
(cid:205)𝑛

𝑊 =
a′ = (𝑎1, 𝑎2, . . . , 𝑎𝑛) = mV−1 (cid:0)mV−1V−1m(cid:1) −1/2
S2 = (cid:205)𝑛

(cid:0)𝑌𝑖 − ¯𝑌(cid:1) 2

𝑖=1

(50)

(51)

Shapiro and Francia [1972] suggested replacing the covariance matrix V by the identity matrix I, because for large
samples, the observations Yi may be treated as if they are independent (see Gupta [1952]). Another asymptotic extension
was suggested by Weisburg and Binham [1975]

𝐸 (𝑋𝑖) = 𝑚𝑖 ≈ Φ−1

(cid:33)

(cid:32) 𝑖 − 3
8
𝑛 + 1
4

𝑖 = 1, 2, . . . , 𝑛

(52)

building atop Elfving [1947]’s approximation but using 3/8 instead of 𝜋/8.

Rahman and Govindarajulu [1997] proposed another variation using the approximation for the expected values of
order statistics given by Blom [1958] and the approximations for the elements of the variance± covariance matrix given
by Blom [1958], Mosteller [2006]. These approximations are

𝐸 (𝑋𝑖) = 𝑚𝑖 ≈ Φ−1

(cid:18)

(cid:19)

,

𝑖
𝑁 + 1

𝑖 = 1, 2, . . . , 𝑛

cov (cid:0)𝑋𝑖 , 𝑋𝑗 (cid:1) = 𝑣𝑖𝑗 ≈

𝑝𝑖 𝑝 𝑗
(𝑛 + 2) 𝑓 (𝑚𝑖) 𝑓 (cid:0)𝑚𝑗 (cid:1)

,

𝑖, 𝑗 = 1, 2, . . . , 𝑛

𝑝𝑖 =

𝑖
𝑛 + 1

We know (see Hammersley and Morton [1954], Plackett [1958])

V−1 = (𝑛 + 1)(𝑛 + 2)

2𝜙2 (𝑚1)
−𝜙 (𝑚1) 𝜙 (𝑚2)
0
...
0

×

(cid:169)
(cid:173)
(cid:173)
(cid:173)
(cid:173)
(cid:173)
(cid:173)
(cid:171)

−𝜙 (𝑚1) 𝜙 (𝑚2)
2𝜙2 (𝑚2)
−𝜙 (𝑚2) 𝜙 (𝑚3)

0
−𝜙 (𝑚2) 𝜙 (𝑚3)
2𝜙2 (𝑚3)

0
0
−𝜙 (𝑚3) 𝜙 (𝑚4)

. . .
. . .
. . .

0
0
0

0

0

0

. . .

2𝜙2 (𝑚𝑛)

(cid:170)
(cid:174)
(cid:174)
(cid:174)
(cid:174)
(cid:174)
(cid:174)
(cid:172)

(53)

(54)

(55)

(56)

F Multivariate Statistics
We ideally would like to compare the distributions. One slight variation is to compare the Characteristic function of the
distributions. Given samples 𝒙1, . . . , 𝒙𝑁 , the Empirical Characteristic Function (ECF) is defined as

ˆ𝜓𝑁 (𝒕) =

1
𝑁

𝑁
(cid:213)

𝑛=1

⊤

𝑒−𝑖𝒕

𝒚𝑛 .

We can now compare our ECF to the one of the target distribution and build the statistic

∫

𝑁

| ˆ𝜓𝑁 (𝒕) − 𝜓0(𝒕)|2𝜔(𝒕)𝑑𝑡 = 𝑁

∫

| ˆ𝜓𝑁 (𝒕) − 𝑒−∥𝒕∥2/2|2𝜔(𝒕)𝑑𝑡,

44

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

if the weighting function is given by 𝜔(𝒕) = (2𝜋𝛽2)−𝑑/2𝑒−

∥𝒕∥2
2
2

then the following simplification can be made

BHEP𝑛,𝛽 =

1
𝑛

𝑛
(cid:213)

𝑗,𝑘=1

(cid:32)

exp

−

𝛽2 (cid:13)

(cid:13)𝑌𝑛,𝑗 − 𝑌𝑛,𝑘
2

(cid:33)

2
(cid:13)
(cid:13)

−

2
(cid:0)1 + 𝛽2(cid:1) 𝑑/2

𝑛
(cid:213)

𝑗=1

(cid:32)

exp

−

(cid:33)

2
𝛽2 (cid:13)
(cid:13)
(cid:13)𝑌𝑛,𝑗
(cid:13)
2 (cid:0)1 + 𝛽2(cid:1)

+

𝑛
(cid:0)1 + 2𝛽2(cid:1) 𝑑/2

.

with 𝛽 > 0, Baringhaus-Henze-Epps-Pulley. From 1 leading to the HZ test 2 uses

the same can be done with the moment generating function 3

𝛽𝑛 = 2−1/2((2𝑑 + 1)𝑛/4)1/(𝑑+4)

(57)

𝑇𝑛,𝛽 = 𝜋𝑑/2 (cid:169)
(cid:173)
(cid:171)

1
𝑛

𝑛
(cid:213)

𝑖,𝑗=1

1
𝛽𝑑/2

exp

(cid:32) (cid:13)
(cid:13)𝑌𝑛,𝑖 + 𝑌𝑛,𝑗
4𝛽

(cid:33)

2
(cid:13)
(cid:13)

+

𝑛
(𝛽 − 1)𝑑/2

−2

𝑛
(cid:213)

𝑗=1

1
(𝛽 − 1/2)𝑑/2

exp

(cid:33)

(cid:32) (cid:13)
2
(cid:13)
(cid:13)𝑌𝑛,𝑗
(cid:13)
4𝛽 − 2

,

(cid:170)
(cid:174)
(cid:172)

here with 𝛽 > 2

There is also one combining both4!

𝑇𝑛,𝛾 := ∫
𝑈𝑛(𝑡) :=

R𝑑 𝑈 2
√

𝑛(𝑡)𝑤𝛾(𝑡)d𝑡

𝑛 (𝑅𝑛(𝑡)𝑀𝑛(𝑡) − 1)

𝑇𝑛,𝛾 =

(cid:18) 𝜋
𝛾

(cid:19) 𝑑/2 


(cid:13)
𝑌+
(cid:13)
(cid:13)
𝑗𝑘

1
2𝑛3

𝑛
(cid:213)

𝑗,𝑘,𝑙,𝑚=1

2
(cid:13)
(cid:13)
(cid:13)

− (cid:13)

(cid:13)𝑌+

ℓ 𝑚

2
(cid:13)
(cid:13)

+ exp (cid:169)
(cid:173)
(cid:173)
(cid:171)
𝑛
(cid:213)

−

2
𝑛

𝑗,𝑘=1

4𝛾

(cid:32) (cid:13)
(cid:13)𝑌𝑛,𝑗

exp

2
(cid:13)
(cid:13)
(cid:13)

(cid:13)
𝑌+
(cid:13)
(cid:13)
𝑗 𝑘



exp (cid:169)

(cid:173)

(cid:173)


(cid:171)

(cid:32) 𝑌+⊤
𝑗 𝑘

cos

𝑌+
ℓ 𝑚

2𝛾

(cid:33)

2
(cid:13)
(cid:13)

cos

(cid:170)
(cid:174)
(cid:174)
(cid:172)
2 − (cid:13)
(cid:13)
(cid:13)
4𝛾

(cid:13)𝑌𝑛,𝑘

− (cid:13)

(cid:13)𝑌−

ℓ 𝑚

4𝛾

cos

(cid:32) 𝑌+⊤
𝑗 𝑘

𝑌−
ℓ 𝑚

(cid:33)

2𝛾

2
(cid:13)
(cid:13)

(cid:170)
(cid:174)
(cid:174)
(cid:172)

(cid:33)






(cid:32) 𝑌⊤
𝑛,𝑗
2𝛾

𝑌𝑛,𝑘

(cid:33)

+ 𝑛

,





and its simplified version

(cid:19) 𝑑/2 √

(cid:101)𝑇𝑛,𝛾 =

(cid:18) 𝜋
𝛾

Also one testing the derivative 5

(cid:101)𝑇𝑛,𝛾 :=

∫

R𝑑

𝑈𝑛(𝑡)𝑤𝛾(𝑡)d𝑡.

exp

(cid:32) (cid:13)
(cid:13)𝑌𝑛,𝑗

(cid:13)𝑌𝑛,𝑘

2 − (cid:13)
(cid:13)
(cid:13)
4𝛾

(cid:33)

2
(cid:13)
(cid:13)

cos

(cid:33)

𝑌𝑛,𝑘

(cid:32) 𝑌⊤
𝑛,𝑗
2𝛾

1
𝑛2

𝑛
(cid:213)

𝑗,𝑘=1

𝑛 (cid:169)
(cid:173)
(cid:171)

− 1(cid:170)
(cid:174)
(cid:172)

∫

HV𝑛,𝛾 := 𝑛

∥∇𝑀𝑛(𝑡) − 𝑡𝑀𝑛(𝑡)∥2

𝑤𝛾(𝑡)d𝑡
(cid:101)

(58)

(59)

(60)

(61)

(62)

1https://www.routledge.com/Density-Estimation-for-Statistics-and-Data-Analysis/Silverman/p/book/9780412246203?srsltid=

AfmBOoodlL-CtlqL0JVC-LcP6mOWw6VTt51_YstdZOW4W3iuicu1VFyg

2https://www.tandfonline.com/doi/abs/10.1080/03610929008830400
3https://arxiv.org/pdf/1711.07199
4https://arxiv.org/pdf/1706.03029
5https://arxiv.org/pdf/1901.03986

45

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

skewness 6:

skewness 7:

HV𝑛,𝛾 =

1
𝑛

(cid:18) 𝜋
𝛾

(cid:19) 𝑑/2

𝑛
(cid:213)

𝑗,𝑘=1

exp (cid:169)
(cid:173)
(cid:173)
(cid:171)

2
(cid:13)
(cid:13)
(cid:13)

(cid:13)
𝑌+
(cid:13)
(cid:13)

𝑛,𝑗,𝑘
4𝛾

(cid:170)
(cid:174)
(cid:174)
(cid:172)

(cid:169)
𝑌⊤
𝑛,𝑗𝑌𝑛,𝑘 −
(cid:173)
(cid:173)
(cid:171)

2
(cid:13)
(cid:13)
(cid:13)

(cid:13)
𝑌+
(cid:13)
(cid:13)

𝑛,𝑗,𝑘
2𝛾

+

𝑑
2𝛾

+

2
(cid:13)
(cid:13)
(cid:13)

(cid:13)
𝑌+
(cid:13)
(cid:13)

𝑛,𝑗,𝑘
4𝛾2

.

(cid:170)
(cid:174)
(cid:174)
(cid:172)

𝑏1,𝑑 =

1
𝑛2

𝑛
(cid:213)

(cid:16)

𝑗,𝑘=1

(cid:17) 3

𝑌⊤
𝑛,𝑗𝑌𝑛,𝑘

(cid:101)𝑏1,𝑑 =

1
𝑛2

𝑛
(cid:213)

𝑗,𝑘=1

𝑌⊤
𝑛,𝑗𝑌𝑛,𝑘

(cid:13)
(cid:13)𝑌𝑛,𝑗

2 (cid:13)
(cid:13)
(cid:13)𝑌𝑛,𝑘
(cid:13)

2
(cid:13)
(cid:13)

which should be 0 for Gaussian and Kurtosis which should be d(d+2)

𝑏2,𝑑 =

1
𝑛

𝑛
(cid:213)

𝑗=1

(cid:13)
(cid:13)𝑌𝑛,𝑗

4
(cid:13)
(cid:13)

(63)

(64)

(65)

(66)

6https://www.jstor.org/stable/2334770
7https://link.springer.com/article/10.1007/s13171-020-00211-6

46

LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

dimension=128, slices=10

dimension=128, slices=100

dimension=1024, slices=100

Figure 16. Reprise of Figure 6 for additional dimensions and number of 1d projections.

47

−2.50.02.5dim1−202dim2originaldata−2.50.02.5dim1VCReg−2.50.02.5dim1ExtendedJarqueBera−2.50.02.5dim1CramerVonMises−2.50.02.5dim1Watson−2.50.02.5dim1AndersonDarling−2.50.02.5dim1EppsPulley−2.50.02.5dim3−202dim4−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim1−202dim2originaldata−2.50.02.5dim1VCReg−2.50.02.5dim1ExtendedJarqueBera−2.50.02.5dim1CramerVonMises−2.50.02.5dim1Watson−2.50.02.5dim1AndersonDarling−2.50.02.5dim1EppsPulley−2.50.02.5dim3−202dim4−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim1−4−2024dim2originaldata−2.50.02.5dim1VCReg−2.50.02.5dim1ExtendedJarqueBera−2.50.02.5dim1CramerVonMises−2.50.02.5dim1Watson−2.50.02.5dim1AndersonDarling−2.50.02.5dim1EppsPulley−2.50.02.5dim3−202dim4−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3−2.50.02.5dim3LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Figure 17. Depiction of the distribution of optimized 𝛽 values from OLS when comparing 𝒁iso and 𝒁aniso from lemmas. 1 and 2. We clearly observe
that the anisotropic version (blue) provides much lower variance compared to the isotropic case (red). We consider a binary classification (linear
separable class) (top row), a linear regression task (middle row), and a nonlinear regression task with smooth targets (bottom row). For each case, we
resample the training samples numerous times and produce an estimate for 𝛽 each time. Because the data is 2-dimensional, we can visualize the 𝛽
distribution directly.

48

0.00.51.01.5β2IsotropicTrueβ0.51.01.5β10.00.51.01.5β20.51.01.5β10.51.01.5β10.51.01.5β1DistributionofEstimatorˆβ(BinaryY)(Normalized)−0.30−0.25−0.20−0.15−0.10β2IsotropicTrueβ−1.05−1.00−0.95−0.90β1−0.30−0.25−0.20−0.15−0.10β2−1.05−1.00−0.95−0.90β1−1.05−1.00−0.95−0.90β1−1.05−1.00−0.95−0.90β1DistributionofEstimatorˆβ(LinearY)02β2Isotropic02β102β202β102β102β1DistributionofEstimatorˆβ(SmoothY)LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Figure 18. Depiction of accuracy (top) and cosine similarity between estimated and true estimator (bottom) for the OLS setting with varying
strength of Tikhonov regularization (x-axis) comparing isotropic and anisotropic embeddings. As per thm. 6, the anisotropic distribution creates a bias
in the OLS estimation for nonzero regularization.

Figure 19. Additional figures provides in Figure 19

49

10−410−310−210−1100101102Regularizationλ0.900.920.940.960.98TestAccuracyTestAccuracyvsRegularization(N=50,LinearClassiﬁcation)Isotropic(κ=1)Anisotropic(κ=5)Anisotropic(κ=10)Anisotropic(κ=50)10−410−310−210−1100101102Regularizationλ0.900.920.940.960.981.00TestAccuracyTestAccuracyvsRegularization(N=200,LinearClassiﬁcation)Isotropic(κ=1)Anisotropic(κ=5)Anisotropic(κ=10)Anisotropic(κ=50)10−410−310−210−1100101102Regularizationλ0.940.950.960.970.980.991.00TestAccuracyTestAccuracyvsRegularization(N=1000,LinearClassiﬁcation)Isotropic(κ=1)Anisotropic(κ=5)Anisotropic(κ=10)Anisotropic(κ=50)10−410−310−210−1100101102Regularizationλ0.60.70.80.91.0E[cos(ˆw,w∗)]DirectionalAlignmentvsRegularization(N=50,LinearRegression)Isotropic(κ=1)Anisotropic(κ=5)Anisotropic(κ=10)Anisotropic(κ=50)10−410−310−210−1100101102Regularizationλ0.750.800.850.900.951.00E[cos(ˆw,w∗)]DirectionalAlignmentvsRegularization(N=200,LinearRegression)Isotropic(κ=1)Anisotropic(κ=5)Anisotropic(κ=10)Anisotropic(κ=50)10−410−310−210−1100101102Regularizationλ0.900.920.940.960.981.00E[cos(ˆw,w∗)]DirectionalAlignmentvsRegularization(N=1000,LinearRegression)Isotropic(κ=1)Anisotropic(κ=5)Anisotropic(κ=10)Anisotropic(κ=50)20406080Testacc.(%)100101Trainloss(log-scale)Spearmancorr.:98.53%(resnet50galaxy10)λ0.040.080.120.160.20406080Testacc.(%)100101Trainloss(log-scale)Spearmancorr.:99.16%(resnet50inet10)λ0.040.080.120.160.200204060Testacc.(%)100101Trainloss(log-scale)Spearmancorr.:94.52%(ViT/base-8inet1k)λ0.040.080.120.160.2020406080Testacc.(%)100101Trainloss(log-scale)Spearmancorr.:97.63%(ViT/s-8galaxy10)λ0.040.080.120.160.2020406080Testacc.(%)100101102Trainloss(log-scale)Spearmancorr.:97.97%(ViT/s-8inet10)λ0.040.080.120.160.20204060Testacc.(%)100Trainloss(log-scale)Spearmancorr.:93.82%(resnet18ﬂowers102)λ0.010.020.050.10.2LeJEPA:
Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments

Figure 20. Proposed trapezoid quadrature for the Epps-Pulley statistic as implemented in algorithm 1. We depict the approximation error of the
integral for various distributions, demonstrate rapid convergence (faster than quadratic show in grey line) across possible embedding distributions.

Figure 21. Additional figures for Figure 10.

50

20406080100Numberofquadraturepoints10−610−510−410−310−210−1Absoluteerror|ˆTn−Ttrue|N(0,1)Trapezoid(Ttrue=1.94)20406080100Numberofquadraturepoints10−11001011021030.5N(−2,0.52)+0.5N(2,0.52)Trapezoid(Ttrue=55109.93)20406080100Numberofquadraturepoints10−310−210−1100101102103Student-t(ν=3)Trapezoid(Ttrue=925.27)100101SIGRegloss(log-scale)10−1100101Pred.loss(log-scale)ViT/s-8-galaxy1010.0027.4744.9462.4179.88Accuracy101SIGRegloss(log-scale)10−1100101Pred.loss(log-scale)ViT/s-8-inet1010.0030.6951.3872.0892.77Accuracy101SIGRegloss(log-scale)10−1100Pred.loss(log-scale)resnet18-ﬂowers10211.0726.3541.6356.9072.18Accuracy