# v3 Exemplars (1 per archetype)

## W10_vague_to_concrete
**Prompt** (discipline: biology, source: synthetic_v3):
> How can we design proteins with more reliable outcomes?

**Skills demonstrated**: problem_clarification, lineage_construction

**Completion** (first 2500 chars):
```
The starting question is too broad—“reliable outcomes” could mean reliable folding, binding, expression, clinical efficacy, or safety—so I first scan the 2018–2025 biology literature landscape to see which protein-adjacent entry points appear in the retrieved corpus.  
```action
{"tool":"search","query":"biology 2018 2025 reliable outcomes protein design structure prediction modeling uncertainty","year_min":2018,"year_max":2025,"k":5}
```
[result]: The broad search returns mixed biology hits: oa:W2804822363, *SWISS-MODEL: homology modelling of protein structures and complexes* (2018), is directly relevant to protein structure reliability; oa:W3004280078 concerns SARS-CoV-2 origin and viral proteins but not design methodology. The remaining candidates—oa:W2805983714 on causal inference, oa:W2912654919 on disease burden, and oa:W3183857127 on dyslipidaemia guidelines—are biology/medicine but not protein design. The clearest narrowing axis is computational protein structure modeling as a reliability bottleneck.

Given that the strongest hit is homology modeling, I narrow from “protein design” to the part of the workflow where unreliable outcomes often arise: designs are proposed from uncertain structural models, especially when templates are weak or local regions are poorly constrained.  
```action
{"tool":"search","query":"homology modelling protein design reliability model quality local uncertainty template coverage 2018 2025","year_min":2018,"year_max":2025,"k":5}
```
[result]: The narrowed search again prioritizes oa:W2804822363, which describes SWISS-MODEL as a mature automated homology-modeling platform and emphasizes model quality estimation. Results cluster around template selection, sequence-template identity, oligomeric complex modeling, and per-model confidence as key determinants of downstream reliability. This suggests a more concrete problem: convert model-quality signals into design constraints rather than treating a model as uniformly trustworthy.

I now focus further on a designable niche: mutational or interface design on homology models of protein complexes, where a single global model score can hide unreliable loops, interfaces, or oligomeric contacts.  
```action
{"tool":"search","query":"protein complex homology model design interface mutations local quality estimate avoid unreliable regions","year_min":2018,"year_max":2025,"k":5}
```
[result]: The relevant thread is still anchored by oa:W2804822363, which covers protein structures and c
...[truncated]
```

## W1_pure_discovery
**Prompt** (discipline: chemistry, source: synthetic_v3):
> How can chemical reactions be made more sustainable without sacrificing efficiency or scope?

**Skills demonstrated**: problem_clarification, lineage_construction, multi_source_synthesis

**Completion** (first 2500 chars):
```
I’ll start broadly to map “sustainable reactions” across green chemistry dimensions—energy input, solvents, catalysts, and feedstocks—while watching for approaches that preserve rate and substrate scope.  
```action
{"tool":"search","query":"sustainable chemical reactions green chemistry efficiency scope catalysis solvents energy 2018 2025 review","year_min":2018,"year_max":2025,"k":5}
```
[result]: The broad search returns mixed sustainability literature, including a non-chemistry CSR review (oa:W3159759907, 2021) that is not mechanistically useful. More relevant chemistry-facing hits include photo-/thermal catalysis (oa:W3111234714, 2020), green nanoparticle synthesis and catalytic/nanomedicine uses (oa:W2911517174, 2019), and bio-derived nanocellulose materials (oa:W2800090145, 2018). The dominant technical themes are renewable-energy-driven catalysis, recyclable heterogeneous catalysts, and safer/biogenic materials.

The broad results suggest energy-input substitution may be the best lever for keeping efficiency while reducing heat/fossil demand, so I’ll narrow to light-driven and hybrid catalytic processes.  
```action
{"tool":"search","query":"photo thermal catalysis sustainable chemical reactions light heat synergy efficiency broad substrate scope review","year_min":2018,"year_max":2025,"k":5}
```
[result]: The search strongly surfaces oa:W3111234714, “Fundamentals and applications of photo-thermal catalysis” (2020), which frames light as both an energetic and electronic promoter for reactions otherwise requiring high temperatures. Results emphasize plasmonic heating, semiconductor photothermal effects, and catalyst designs that combine thermal conversion with photo-induced charge carriers. Reported applications include CO2 conversion, hydrogen production, VOC oxidation, and selective organic transformations, but selectivity/scope control remains uneven.

To avoid a proposal that is just “use light,” I’ll look for sustainable catalyst/support materials that could make photothermal chemistry recyclable, low-toxicity, and compatible with diverse substrates.  
```action
{"tool":"search","query":"biobased recyclable catalyst support nanocellulose metal nanoparticles sustainable organic reactions 2018 2025","year_min":2018,"year_max":2025,"k":5}
```
[result]: The search returns oa:W2800090145 on nanocellulose as a renewable high-surface-area material and oa:W2911517174 on silver nanoparticles, including synthesis routes and application contexts. Together
...[truncated]
```

## W2_single_paper_extension
**Prompt** (discipline: biology, source: synthetic_v3):
> Given a recent paper presenting a technique to profile metabolic activity across heterogeneous tumor regions, what is the natural extension to ask whether metabolic niches influence therapy response or metastatic potential?

**Skills demonstrated**: lineage_construction, gap_identification, reflection_revision

**Completion** (first 2500 chars):
```
I first extract the anchor’s “genome” to identify the method, biological niche, observations, and—most importantly—the stated limitation that should drive a single-paper extension.  
```action
{"tool":"extract_genome","paper_id":"oa:W4220662588"}
```
[result]: The anchor, **INTEGRATE: Model-based multi-omics data integration to characterize multi-level metabolic regulation** (oa:W4220662588, 2022), contributes a model-based framework for inferring metabolic activity/regulation from multi-omics measurements. Its genome centers on resolving heterogeneous metabolic regulatory states rather than simply measuring metabolite abundance. The key limitation is that inferred metabolic niches/states are largely descriptive and cross-sectional: they are not directly tested for whether they predict therapy response, metastatic competence, or functional tumor behavior.

To ground the extension, I search for nearby work connecting metabolic regulation, tissue/tumor heterogeneity, and phenotypic consequences such as response or dissemination.  
```action
{"tool":"search","query":"tumor metabolic heterogeneity metabolic niches therapy response metastasis multi-omics spatial metabolism","year_min":2018,"year_max":2025,"k":5}
```
[result]: The search returns oa:W4220662588 as the closest metabolic-regulation methods anchor, plus adjacent but less directly relevant works from the supplied candidate set. oa:W4281625009 discusses direct cell-cell communication routes that could mediate niche effects across tumor regions, while oa:W3023972366 and oa:W3133958426 are largely outside the tumor-metabolism niche. Overall, the gap remains: profiling metabolic states is available, but linking spatial/metabolic niches to therapy response or metastatic potential is not directly addressed.

I read the anchor-adjacent communication paper because metastatic potential and therapy resistance often depend on interactions among tumor cells, stromal cells, immune cells, and metabolically specialized regions.  
```action
{"tool":"read","paper_id":"oa:W4281625009"}
```
[result]: **Direct Cell-Cell Communication via Membrane Pores, Gap Junction Channels, and Tunneling Nanotubes** (oa:W4281625009, 2022) reviews mechanisms by which cells exchange ions, metabolites, organelles, and stress signals. It suggests that metabolic phenotypes may propagate across heterogeneous tissue through direct communication, not only through soluble gradients. This supports an extension in which metabolic niches are trea
...[truncated]
```

## W3_multi_paper_synthesis
**Prompt** (discipline: astronomy, source: synthetic_v3):
> Spectral line diagnostics, dust-continuum observations, and chemical evolution models all attack different aspects of star-forming regions; how can they be combined to understand how local environments shape the birth of stars?

**Skills demonstrated**: multi_source_synthesis, method_transfer

**Completion** (first 2500 chars):
```
I first verify the retrieved OpenAlex candidate and use it as the seed object, even though the title suggests it may be a noisy proceedings-level record rather than a clean research article. This matters because the synthesis should not overfit to a bibliographic artifact if the useful content is only abstract-level.

```action
{"tool": "read", "paper_id": "oa:W7112541710"}
```

[result]: The record `oa:W7112541710` is a proceedings/book-of-abstracts style item rather than a single focused astronomy paper. It contains heterogeneous seminar abstracts, so it is weak as a direct anchor but useful as a signal that the retrieved topic space is broad and interdisciplinary. I will treat it as a noisy seed and extract only the astronomy-relevant genome implied by the prompt: line diagnostics, dust continuum, and chemical/evolutionary modeling of star-forming environments.

I extract the seed genome from the provided OpenAlex record, focusing on the implied mechanism families rather than the proceedings container itself. The goal is to identify the high-level components that can be compared against more targeted observational and modeling genomes.

```action
{"tool": "extract_genome", "paper_id": "oa:W7112541710"}
```

[result]: The extracted genome is broad and weakly specified: it links local star-forming environments to observables such as emission-line tracers, dust continuum structure, and chemical state. Its mechanism is “multi-probe environmental inference,” but the limitation is that the record does not define a concrete integration pipeline. The useful claim space is that star formation must be interpreted through coupled physical, chemical, and radiative diagnostics rather than any single tracer.

To obtain the missing targeted anchors, I search within 2018–2025 for work combining dense-gas lines, dust continuum, and astrochemical or evolutionary interpretation in star-forming regions. This step supplies the observational/modeling poles needed for pairwise genome comparison.

```action
{"tool": "search", "query": "2018 2025 star forming regions spectral line diagnostics dust continuum chemical evolution models local environment", "year_min": 2018, "year_max": 2025, "k": 5}
```

[result]: The search returns three useful anchor classes: ALMA/NOEMA studies using molecular-line diagnostics of dense gas kinematics and excitation; dust-continuum surveys mapping cores, filaments, and mass reservoirs; and chemical-evolution models tracking abundance clocks, deple
...[truncated]
```

## W4_literature_review
**Prompt** (discipline: astro, source: gene_arena):
> You are a senior researcher in Astrophysics / Computational Cosmology.
Below are 7 important recent papers in this area, presented in random order. Each item contains only the paper title and a brief abstract; no lineage order, structured source-change fields, or evolutionary dynamics labels are provided.

**Frontier Question**:
The weak-lensing simulation-based-inference benchmark shows that field-level posterior estimation can be standardized, but its simplified log-normal simulator omits the 

**Skills demonstrated**: multi_source_synthesis, problem_clarification

**Completion** (first 2500 chars):
```
I first map the benchmark paper to the broader weak-lensing SBI literature, looking for methods that explicitly compare full-field neural inference against likelihood-based baselines.  
```action
{"tool": "search", "query": "\"Simulation-based inference benchmark\" \"weak lensing\" cosmology sbi_lens", "year_min": 2018, "year_max": 2025, "k": 5}
```  
[result]: The search returns the recent benchmark paper “Simulation-based inference benchmark for weak lensing cosmology” (OpenAlex: `W4391136188`) and related LSST-Y10 log-normal field-level SBI studies. Results emphasize that the benchmark standardizes posterior metrics, calibration tests, and map-level neural likelihood/posterior estimators, but uses deliberately simplified convergence-map simulations.

To ground the systems question, I search for baryonic-feedback weak-lensing papers that quantify the scale and parameter bias induced by astrophysical uncertainty.  
```action
{"tool": "search", "query": "\"Modeling baryonic physics\" \"future weak lensing surveys\" hydrodynamical simulations", "year_min": 2018, "year_max": 2025, "k": 5}
```  
[result]: The key hit is “Modeling baryonic physics in future weak lensing surveys” (OpenAlex: `W2566511904`), frequently cited in later LSST/Euclid/Roman systematics discussions. Follow-up results include baryonification, HMcode, and hydrodynamical-emulator approaches showing that baryons create percent-level to tens-of-percent changes in small-scale lensing power and can strongly bias Ωm, σ8/S8, and dark-energy constraints.

Next I search from the simulation-suite side, because calibrated field-level inference needs nuisance-parameter coverage rather than a single forward model.  
```action
{"tool": "search", "query": "CAMELS project Cosmology Astrophysics Machine Learning Simulations weak lensing inference baryonic feedback", "year_min": 2018, "year_max": 2025, "k": 5}
```  
[result]: The search identifies “The CAMELS project: Cosmology and Astrophysics with MachinE Learning Simulations” (OpenAlex: `W3169027791`) as a foundational simulation ensemble spanning cosmological and feedback parameters. Related papers use CAMELS for neural compression, likelihood-free inference, robustness tests, and cross-simulation transfer, making it central to the question of amortized inference under baryonic uncertainty.

Finally I search for the machine-learning/SBI machinery that can connect such simulation ensembles to calibrated posterior estimation.  
```action
{"tool": "search
...[truncated]
```

## W5_critical_analysis
**Prompt** (discipline: astronomy, source: synthetic):
> Research topic: Measure the expansion history of the universe using Type Ia supernovae, baryon acoustic oscillations, and standard sirens to test possible deviations from the standard ΛCDM cosmological model.

Discipline: astronomy

Propose a novel follow-up research idea building on relevant prior work.

**Skills demonstrated**: critical_evaluation, hypothesis_specification

**Completion** (first 2500 chars):
```
I start from a recent anchor that explicitly combines low-redshift Type Ia supernova distances with BAO/CMB-era expansion constraints, because it is a natural baseline for testing ΛCDM deviations before adding standard sirens. Reading it first lets the later critique focus on what, exactly, the prior work assumed about distance calibration and the dark-energy parameterization.  
```action
{"tool": "read", "paper_id": "https://openalex.org/W4297420530"}
```  
[result]: Read Brout et al. 2022, “The Pantheon+ Analysis: Cosmological Constraints” (ApJ), a major compilation of 1701 light curves from 1550 spectroscopically confirmed SNe Ia. The paper constrains ΩM, w, and H0-related combinations using Pantheon+ with external probes such as BAO and CMB priors, finding broad consistency with flat ΛCDM in many combinations but also highlighting dependence on calibration and external datasets. Its likelihood machinery is a strong anchor for a follow-up because it exposes correlated supernova systematics rather than treating SN distances as independent points.

Next I extract the “genome” of the anchor: mechanism, niche, observations, limitations, delta, and claim. This turns the paper into structured ingredients that can be compared against later BAO and standard-siren work.  
```action
{"tool": "extract_genome", "paper_id": "https://openalex.org/W4297420530"}
```  
[result]: The mechanism genome is: calibrated SN Ia Hubble diagram + covariance-aware light-curve standardization + external absolute/relative distance anchors to infer late-time expansion. The niche is late-universe distance-redshift cosmology over roughly 0.01 < z < 2.3, mainly sensitive to ΩM, w, curvature, and H0 degeneracies. The key limitation genome includes SN population drift, photometric calibration correlations, sparse high-z leverage, and degeneracy between absolute calibration and the sound-horizon scale when BAO/CMB information is added. The claim genome is that Pantheon+ improves statistical/systematic precision and remains broadly compatible with ΛCDM under common model choices, while leaving room for tensions in combined-probe fits.

Before looking for critiques, I check whether the core idea—using SNe Ia + BAO + standard sirens together to probe ΛCDM deviations—has already been framed as a solved method in 2018–2025. This identifies whether novelty should come from adding sirens, changing the model, or changing the validation methodology.  
```action
{"tool": "novelty_check", "mechanism"
...[truncated]
```

## W6_cross_domain_bridge
**Prompt** (discipline: chemistry, source: synthetic_v3):
> Can strategies used in ecology to track population shifts be adapted to study how catalytic systems evolve under changing reaction conditions?

**Skills demonstrated**: method_transfer, multi_source_synthesis

**Completion** (first 2500 chars):
```
I first search for an ecology-side method that explicitly follows population/community change under shifting pressures, because the transferable “tracking” logic should come from a domain where adaptive populations are the object of study.  
```action
{"tool":"search","query":"ecology strategies track population shifts eco-evolutionary dynamics biological control changing environments 2018 2025","year_min":2018,"year_max":2025,"k":5}
```
[result]: The search surfaces oa:W3207672482, “Biological Control of Plant Diseases: An Evolutionary and Eco-Economic Consideration” (2021), as a strong ecology candidate because it frames biological control as a dynamic interaction among pest/pathogen, control agent, host, and environment. It also returns oa:W3165326400 on Bacteroidetes polysaccharide degradation and oa:W3005821439 on motile active matter, both relevant to adaptive communities but less directly focused on applied population-shift tracking.

I read the ecology candidate to identify the operational strategy rather than just the biological context.  
```action
{"tool":"read","paper_id":"oa:W3207672482"}
```
[result]: The paper argues that biological control outcomes depend on eco-evolutionary feedbacks: population densities, trait variation, resistance, local adaptation, dispersal, and economic intervention all co-determine whether control succeeds. Its practical stance is that one should monitor not just endpoint suppression, but shifts in interacting populations and traits under changing selection pressures. This gives a candidate transferable method: longitudinal, multi-population state tracking coupled to adaptive-management decisions.

I extract the “genome” of the ecology method: its mechanism, niche, observations, limits, deltas, and claims in a structured form for later transfer.  
```action
{"tool":"extract_genome","paper_id":"oa:W3207672482"}
```
[result]: Mechanism genome: track coupled population abundances and adaptive traits across interacting ecological actors, then update intervention strategy as the system evolves. Niche genome: biological control and plant disease management where outcomes change over seasons, environments, and resistance dynamics. Observation genome: repeated population/trait measurements reveal regime shifts, loss of control, emergence of resistance, and context-dependent success. Limitation genome: ecological systems are noisy, multicausal, and hard to experimentally perturb; causal inference often requires careful longi
...[truncated]
```

## W7_hypothesis_refinement
**Prompt** (discipline: materials, source: synthetic_v3):
> Would mixing recycled materials into cement-like composites improve sustainability while keeping mechanical performance acceptable? The hypothesis is that some waste-derived particles could act as useful fillers, but the effects on durability and processing are unknown.

**Skills demonstrated**: reflection_revision, hypothesis_specification

**Completion** (first 2500 chars):
```
I start by locating recent review and experimental literature that spans “waste-derived fillers” broadly enough to avoid prematurely committing to one recycled stream. The goal is to anchor the rough filler hypothesis in the 2018–2025 cement-composite evidence base.

```action
{"tool":"search","query":"2018 2025 recycled waste particles fillers cementitious composites mechanical durability processing review","year_min":2018,"year_max":2025,"k":5}
```

[result]: The search returned recent cementitious-composite work on recycled glass powder, recycled concrete fines, rubber/plastic particles, and mixed construction-demolition wastes. Highly relevant hits included OpenAlex records `W4389253549` on recycled waste glass in cement-based materials, `W3046689641` on recycled aggregate concrete durability, and `W4392691067` on plastic waste in cementitious composites. The evidence suggests that fine mineral-like wastes can sometimes improve packing or pozzolanic response, while polymeric/rubber wastes often reduce strength but may improve toughness or density-related properties.

Given that initial scan, I formulate a deliberately broad v1 hypothesis: recycled particles can serve as fillers if size and chemistry are selected to preserve packing and interfacial bonding. I leave durability and processing as explicit weak spots to be refined rather than overclaiming.

```action
{"tool":"propose","gene_genome":{"mechanism_genome":"Waste-derived particles added to cement-like composites can improve sustainability by replacing virgin binder or aggregate; finely ground mineral-rich wastes act mainly through filler packing and possible pozzolanic/nucleation effects, while non-mineral particles may act as lightweight or crack-bridging inclusions.","niche_genome":"Cementitious mortars/concretes or alkali-activated cement-like composites using recycled glass, concrete fines, ceramic fines, plastic, or rubber particles as partial replacement of cement, sand, or filler in 2018–2025 sustainable construction materials research.","observation_genome":"Many studies report that low-to-moderate replacement levels can maintain acceptable compressive strength, especially when particles are fine and well graded; larger or hydrophobic particles often reduce workability and strength.","limitation_genome":"The rough hypothesis does not yet distinguish which recycled streams are beneficial versus merely tolerated, nor does it specify durability risks such as ASR, permeability, freeze-thaw, 
...[truncated]
```

## W8_reproduction_doubt
**Prompt** (discipline: energy, source: synthetic_v3):
> Some recent work suggests that demand response can replace a large share of peaking generation, but the assumptions about user behavior seem optimistic. How can we test whether this claim remains valid in more realistic settings?

**Skills demonstrated**: critical_evaluation, hypothesis_specification

**Completion** (first 2500 chars):
```
I start by reading an anchor paper with a strong, policy-relevant claim: flexible demand can substitute for a substantial amount of peaking capacity. I want the exact behavioral and reliability assumptions before looking for failures or replications.  
```action
{"tool":"read","paper_id":"https://openalex.org/W2964088836"}
```
[result]: The anchor study models residential/commercial demand response as a dispatchable capacity resource in a high-renewables grid and finds large reductions in gas peaker utilization and capacity needs. Its core result depends on high enrollment, low event opt-out, fast load recovery, and aggregator control that is treated as dependable during scarcity hours. The paper acknowledges that elasticity and participation are drawn from pilot-scale or program-average evidence rather than from repeated critical-peak events under heat-wave stress.

Next I search specifically for replication, validation, or field-trial evidence on whether demand response actually delivers dependable peak capacity. The concern is not whether DR can shift load once, but whether it behaves like firm capacity across correlated system-stress events.  
```action
{"tool":"search","query":"replication validation demand response peaking capacity reliability field trial opt out critical peak pricing","year_min":2018,"year_max":2025,"k":5}
```
[result]: The search returns field and quasi-experimental studies of critical peak pricing, smart thermostat demand response, and emergency DR programs. Several sources report statistically significant reductions in peak load, but the magnitude varies strongly with weather, baseline methods, customer fatigue, and rebound. The most relevant papers include a large-scale smart-thermostat DR evaluation, a review of residential DR persistence, and a capacity-accreditation paper emphasizing uncertainty and coincidence with peak risk.

I read a field-evaluation paper because it tests the main behavioral assumption directly: do enrolled customers provide load relief during real peak events? This is the closest evidence to a replication of the anchor model’s behavioral module.  
```action
{"tool":"read","paper_id":"https://openalex.org/W3015924217"}
```
[result]: The field study estimates causal peak reductions from thermostat-based residential demand response during summer events. It finds meaningful average reductions, but the distribution is skewed: a minority of homes deliver most savings, reductions degrade during long or repeated
...[truncated]
```

## W9_tool_heavy_comparison
**Prompt** (discipline: astronomy, source: synthetic_v3_supp):
> How do manual inspection, automated classification, and hybrid human-machine approaches compare for discovering rare objects in astronomical surveys?

**Skills demonstrated**: multi_source_synthesis, critical_evaluation

**Completion** (first 2500 chars):
```
I’ll first search across the 2018–2025 window for papers representing three competing discovery modes: visually/manual inspection, automated ML classification/anomaly detection, and hybrid human-in-the-loop systems for astronomical survey rare-object discovery. This gives us candidate methods to read and extract into comparable “method genomes.”  
```action
{"tool": "search", "query": "astronomical surveys rare object discovery manual inspection automated classification human in the loop anomaly detection 2018 2025", "year_min": 2018, "year_max": 2025, "k": 5}
```  
[result]: The search returns three highly relevant method exemplars: W3032120920, “Unexpected circular radio objects at high Galactic latitude” / ORC discovery by visual inspection of ASKAP-EMU images; W3091602921, “Astronomaly: Personalised active anomaly detection in astronomical data”; and W4280207479, “The Zwicky Transient Facility Bright Transient Survey. II. A public statistical sample for exploring supernova demographics,” which uses systematic broker/photometric/spectroscopic classification workflows. It also surfaces W3044314048 on Fink/LSST-style automated alert classification and W2963483542 on convolutional approaches for astronomical source classification. These span manual, hybrid, and automated survey-scale discovery pathways.  

I’ll read the manual-inspection exemplar first because it is the clearest case where rare objects emerged from human visual scrutiny rather than pre-specified classifiers. The goal is to identify what manual inspection contributes—especially sensitivity to visually odd morphologies—and where it fails at scale.  
```action
{"tool": "read", "paper_id": "W3032120920"}
```  
[result]: W3032120920 reports the discovery of odd radio circles in ASKAP/EMU Pilot Survey images after visual inspection of radio-continuum maps. The objects were rare, extended, circular, low-surface-brightness radio features not matching common catalog classes, and their recognition depended on human pattern perception across images. The paper emphasizes novelty and astrophysical ambiguity, but the method is non-systematic, difficult to quantify for completeness, and hard to scale to survey volumes.  

Next I’ll read a representative automated classification stream, focusing on transient-survey pipelines that process very large alert streams and classify candidates by features, light curves, and follow-up labels. This captures high-throughput discovery but may suppress rare or out-of-
...[truncated]
```
