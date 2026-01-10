---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:6170503
- loss:MultipleNegativesRankingLoss
base_model: cross-encoder/ms-marco-TinyBERT-L2-v2
datasets:
- allenai/scirepeval
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-TinyBERT-L2-v2
  results:
  - task:
      type: cross-encoder-classification
      name: Cross Encoder Classification
    dataset:
      name: scidocs cite eval
      type: scidocs_cite_eval
    metrics:
    - type: accuracy
      value: 0.8223
      name: Accuracy
    - type: accuracy_threshold
      value: 0.5697983503341675
      name: Accuracy Threshold
    - type: f1
      value: 0.8340599202300343
      name: F1
    - type: f1_threshold
      value: 0.41046997904777527
      name: F1 Threshold
    - type: precision
      value: 0.7777201176267082
      name: Precision
    - type: recall
      value: 0.8992
      name: Recall
    - type: average_precision
      value: 0.879899419455878
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-TinyBERT-L2-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-TinyBERT-L2-v2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2) on the [scirepeval](https://huggingface.co/datasets/allenai/scirepeval) dataset using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-TinyBERT-L2-v2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2) <!-- at revision 81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Training Dataset:**
    - [scirepeval](https://huggingface.co/datasets/allenai/scirepeval)
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['Detecting tax evasion: a co-evolutionary approach [SEP] We present an algorithm that can anticipate tax evasion by modeling the co-evolution of tax schemes with auditing policies. Malicious tax non-compliance, or evasion, accounts for billions of lost revenue each year. Unfortunately when tax administrators change the tax laws or auditing procedures to eliminate known fraudulent schemes another potentially more profitable scheme takes it place. Modeling both the tax schemes and auditing policies within a single framework can therefore provide major advantages. In particular we can explore the likely forms of tax schemes in response to changes in audit policies. This can serve as an early warning system to help focus enforcement efforts. In addition, the audit policies can be fine tuned to help improve tax scheme detection. We demonstrate our approach using the iBOB tax scheme and show it can capture the co-evolution between tax evasion and audit policy. Our experiments shows the expected oscillatory behavior of a biological co-evolving system.', 'An Intelligent Anti-Money Laundering Method for Detecting Risky Users in the Banking Systems [SEP] During the last decades, universal economy has experienced money laundering and its destructive impact on the economy of the countries. Money laundering is the process of converting or transferring an asset in order to conceal its illegal source or assist someone that is involved in such crimes. Criminals generally attempt to clean the sources of the funds obtained by crime, using the banking system. Due to the large amount of information in the banks, detecting such behaviors is not feasible without anti-money laundering systems. Money laundering detection is one of the areas, where data mining tools can be useful and effective. In this research, some of the features of the users are extracted from their profiles by studying them. These features may include large financial transactions in risky areas regarding money laundering, reactivation of dormant accounts with considerable amounts, etc. Network training is performed by designing a fuzzy system, developing an adaptive neuro-fuzzy inference system and adding feature vectors of the users to it. The network output can determine the riskiness of the user behavior. The evaluation results reveal that the proposed method increases the accuracy of detecting risky users.'],
    ['Lethal effects of abamectin on the aquatic organisms Daphnia similis, Chironomus xanthus and Danio rerio. [SEP] Abamectin is used as an acaricide and insecticide for fruits, vegetables and ornamental plants, as well as a parasiticide for animals. One of the major problems of applying pesticides to crops is the likelihood of contaminating aquatic ecosystems by drift or runoff. Therefore, toxicity tests in the laboratory are important tools to predict the effects of chemical substances in aquatic ecosystems. The aim of this study was to assess the potential hazards of abamectin to the freshwater biota and consequently the possible losses of ecological services in contaminated water bodies. For this purpose, we identified the toxicity of abamectin on daphnids, insects and fish. Abamectin was highly toxic, with an EC(50) 48 h for Daphnia similis of 5.1 ng L(-1), LC(50) 96 h for Chironomus xanthus of 2.67 μg L(-1) and LC(50) 48 h for Danio rerio of 33 μg L(-1).', 'Effects of in vitro and in vivo avermectin exposure on alpha synuclein expression and proteasomal activity in pigeons. [SEP] Avermectins (AVMs) are used worldwide in agriculture and veterinary medicine. Residues of avermectin drugs, causing toxicological effects on non-target organisms, have raised great concern. The aim of this study was to investigate the effects of AVM on the expression levels of alpha synuclein (α-Syn) and proteasomal activity in pigeon (Columba livia) neurons both in vivo and in vitro. The results showed that, the mRNA and protein levels of α-Syn increased in AVM treated groups relative to control groups in the cerebrum, cerebellum and optic lobe in vivo. Dose-dependent decreases in the proteasomal activity (i.e., chymotrypsin-like, trypsin-like and peptidylglutamyl peptidehydrolase) were observed both in vivo and in vitro. The results suggested that AVM could induce the expression levels of α-Syn and inhibit the normal physiological function of proteasome in brain tissues and neurons. The information presented in this study is helpful to understand the mechanism of AVM-induced neurotoxicology in birds.'],
    ['IGF1 activates cell cycle arrest following irradiation by reducing binding of ΔNp63 to the p21 promoter [SEP] Radiotherapy for head and neck tumors often results in persistent loss of function in salivary glands. Patients suffering from impaired salivary function frequently terminate treatment prematurely because of reduced quality of life caused by malnutrition and other debilitating side-effects. It has been previously shown in mice expressing a constitutively active form of Akt (myr-Akt1), or in mice pretreated with IGF1, apoptosis is suppressed, which correlates with maintained salivary gland function measured by stimulated salivary flow. Induction of cell cycle arrest may be important for this protection by allowing cells time for DNA repair. We have observed increased accumulation of cells in G2/M at acute time-points after irradiation in parotid glands of mice receiving pretreatment with IGF1. As p21, a transcriptional target of the p53 family, is necessary for maintaining G2/M arrest, we analyzed the roles of p53 and p63 in modulating IGF1-stimulated p21 expression. Pretreatment with IGF1 reduces binding of ΔNp63 to the p21 promoter after irradiation, which coincides with increased p53 binding and sustained p21 transcription. Our data indicate a role for ΔNp63 in modulating p53-dependent gene expression and influencing whether a cell death or cell cycle arrest program is initiated.', 'ΔNp63 expression in four carcinoma cell lines and the effect on radioresistance—a siRNA knockdown model [SEP] ObjectivesThis study investigated the expression of ΔNp63α in carcinoma cell lines of the upper aerodigestive tract and their potential influence on radioresistance using a small interfering RNA (siRNA) knockdown approach.Materials and methodsFour carcinoma cell lines were investigated for the expression of the ΔNp63 isoform by quantitative reverse transcriptase polymerase chain reaction (qRT-PCR) (0, 24, 48\xa0h) with and without single dose irradiation of 6\xa0Gy. Furthermore, all cell lines were transfected with siRNA against the ΔNp63α isoform over 24\xa0h. Knockdown effectiveness was controlled by qRT-PCR and Western blot. Apoptotic events were evaluated by terminal transferase dUTP nick end labeling (TUNEL) assay and cross-checked by a test for cell viability (WST-1, Roche) over 48\xa0h.ResultsAll cell lines presented varying expression of the ΔNp63α isoform with and without irradiation. A sufficient knockdown rate was established by siRNA transfection. Knockdown of the ΔNp63 isoform showed an effect on radiation sensitivity proven by an increase of apoptotic events detectable by immunofluorescence (TUNEL assay) and likewise a significant reduction of formazan production (WST-1 test) in three cell lines.ConclusionsWe found overexpression of ΔNp63α with and without irradiation in three cell lines, and the knockdown of ΔNp63α led to increased apoptotic events and fewer viable cells. Thus, the overexpression of ΔNp63α might protect carcinoma cells against irradiation effects.Clinical relevanceThe present work supports the hypothesis that protein 63 might serve as a negative predictor for irradiation response and survival in a clinical setting and may be a target for future therapeutic strategies.'],
    ['InAs migration on released, wrinkled InGaAs membranes used as virtual substrate [SEP] Partly released, relaxed and wrinkled InGaAs membranes are used as virtual substrates for overgrowth with InAs. Such samples exhibit different lattice parameters for the unreleased epitaxial parts, the released flat, back-bond areas and the released wrinkled areas. A large InAs migration towards the released membrane is observed with a material accumulation on top of the freestanding wrinkles during overgrowth. A semi-quantitative analysis of the misfit strain shows that the material migrates to the areas of the sample with the lowest misfit strain, which we consider as the areas of the lowest chemical potential of the surface. Material migration is also observed for the edge-supported, freestanding InGaAs membranes found on these samples. Our results show that the released, wrinkled nanomembranes offer a growth template for InAs deposition that fundamentally changes the migration behavior of the deposited material on the growth surface.', 'In adatom diffusion on InxGa1−xAs/GaAs(001): effects of strain, reconstruction and composition [SEP] By using density functional theory (DFT) calculations of the potential energy surface in conjunction with the analytical solution of the master equation for the time evolution of the adatom site distribution, we study the diffusion properties of an isolated In adatom on InxGa1−xAs wetting layers (WL) deposited on the GaAs(001). The WL reconstructions considered in this study are, listed in the order of increasing In coverage: c(4 × 4), (1 × 3), (2 × 3), α2(2 × 4) and β2(2 × 4). We analyze the dependence of the diffusion properties on WL reconstruction, composition and strain, and find that: (i) diffusion on the (2 × N) reconstructions is strongly anisotropic, owing to the presence of the low barrier potential in-dimer trench, favoring the diffusion along the direction over that along the [110] direction; (ii) In diffusion at a WL coverage θ = 2/3 monolayers (ML; with composition x = 2/3) is faster than on clean GaAs(001) c(4 × 4), and decreases at θ = 1.75 ML (x = 1; e.g. InAs/GaAs(001)); (iii) diffusion and nucleation on the (2 × 4) WL is affected by the presence of adsorption sites for indium inside the As dimers; (iv) the approximation used for the exchange–correlation potential within DFT has an important effect on the description of the diffusion properties.'],
    ['Corpus building for Mongolian language [SEP] This paper presents an ongoing research aimed to build the first corpus, 5 million words, for Mongolian language by focusing on annotating and tagging corpus texts according to TEI XML (McQueen, 2004) format. Also, a tool, MCBuilder, which provides support for flexibly and manually annotating and manipulating the corpus texts with XML structure, is presented.', 'Mongolian speech corpus for text-to-speech development [SEP] This paper presents a first attempt to develop Mongolian speech corpus that designed for data-driven speech synthesis in Mongolia. The aim of the speech corpus is to develop a high-quality Mongolian TTS for blinds to use with screen reader. The speech corpus contains nearly 6 hours of Mongolian phones. It well provides Cyrillic text transcription and its phonetic transcription with stress marking. It also provides context information including phone context, stressing levels, syntactic position in word, phrase and utterance for modeling speech acoustics and characteristics for speech synthesis.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Detecting tax evasion: a co-evolutionary approach [SEP] We present an algorithm that can anticipate tax evasion by modeling the co-evolution of tax schemes with auditing policies. Malicious tax non-compliance, or evasion, accounts for billions of lost revenue each year. Unfortunately when tax administrators change the tax laws or auditing procedures to eliminate known fraudulent schemes another potentially more profitable scheme takes it place. Modeling both the tax schemes and auditing policies within a single framework can therefore provide major advantages. In particular we can explore the likely forms of tax schemes in response to changes in audit policies. This can serve as an early warning system to help focus enforcement efforts. In addition, the audit policies can be fine tuned to help improve tax scheme detection. We demonstrate our approach using the iBOB tax scheme and show it can capture the co-evolution between tax evasion and audit policy. Our experiments shows the expected oscillatory behavior of a biological co-evolving system.',
    [
        'An Intelligent Anti-Money Laundering Method for Detecting Risky Users in the Banking Systems [SEP] During the last decades, universal economy has experienced money laundering and its destructive impact on the economy of the countries. Money laundering is the process of converting or transferring an asset in order to conceal its illegal source or assist someone that is involved in such crimes. Criminals generally attempt to clean the sources of the funds obtained by crime, using the banking system. Due to the large amount of information in the banks, detecting such behaviors is not feasible without anti-money laundering systems. Money laundering detection is one of the areas, where data mining tools can be useful and effective. In this research, some of the features of the users are extracted from their profiles by studying them. These features may include large financial transactions in risky areas regarding money laundering, reactivation of dormant accounts with considerable amounts, etc. Network training is performed by designing a fuzzy system, developing an adaptive neuro-fuzzy inference system and adding feature vectors of the users to it. The network output can determine the riskiness of the user behavior. The evaluation results reveal that the proposed method increases the accuracy of detecting risky users.',
        'Effects of in vitro and in vivo avermectin exposure on alpha synuclein expression and proteasomal activity in pigeons. [SEP] Avermectins (AVMs) are used worldwide in agriculture and veterinary medicine. Residues of avermectin drugs, causing toxicological effects on non-target organisms, have raised great concern. The aim of this study was to investigate the effects of AVM on the expression levels of alpha synuclein (α-Syn) and proteasomal activity in pigeon (Columba livia) neurons both in vivo and in vitro. The results showed that, the mRNA and protein levels of α-Syn increased in AVM treated groups relative to control groups in the cerebrum, cerebellum and optic lobe in vivo. Dose-dependent decreases in the proteasomal activity (i.e., chymotrypsin-like, trypsin-like and peptidylglutamyl peptidehydrolase) were observed both in vivo and in vitro. The results suggested that AVM could induce the expression levels of α-Syn and inhibit the normal physiological function of proteasome in brain tissues and neurons. The information presented in this study is helpful to understand the mechanism of AVM-induced neurotoxicology in birds.',
        'ΔNp63 expression in four carcinoma cell lines and the effect on radioresistance—a siRNA knockdown model [SEP] ObjectivesThis study investigated the expression of ΔNp63α in carcinoma cell lines of the upper aerodigestive tract and their potential influence on radioresistance using a small interfering RNA (siRNA) knockdown approach.Materials and methodsFour carcinoma cell lines were investigated for the expression of the ΔNp63 isoform by quantitative reverse transcriptase polymerase chain reaction (qRT-PCR) (0, 24, 48\xa0h) with and without single dose irradiation of 6\xa0Gy. Furthermore, all cell lines were transfected with siRNA against the ΔNp63α isoform over 24\xa0h. Knockdown effectiveness was controlled by qRT-PCR and Western blot. Apoptotic events were evaluated by terminal transferase dUTP nick end labeling (TUNEL) assay and cross-checked by a test for cell viability (WST-1, Roche) over 48\xa0h.ResultsAll cell lines presented varying expression of the ΔNp63α isoform with and without irradiation. A sufficient knockdown rate was established by siRNA transfection. Knockdown of the ΔNp63 isoform showed an effect on radiation sensitivity proven by an increase of apoptotic events detectable by immunofluorescence (TUNEL assay) and likewise a significant reduction of formazan production (WST-1 test) in three cell lines.ConclusionsWe found overexpression of ΔNp63α with and without irradiation in three cell lines, and the knockdown of ΔNp63α led to increased apoptotic events and fewer viable cells. Thus, the overexpression of ΔNp63α might protect carcinoma cells against irradiation effects.Clinical relevanceThe present work supports the hypothesis that protein 63 might serve as a negative predictor for irradiation response and survival in a clinical setting and may be a target for future therapeutic strategies.',
        'In adatom diffusion on InxGa1−xAs/GaAs(001): effects of strain, reconstruction and composition [SEP] By using density functional theory (DFT) calculations of the potential energy surface in conjunction with the analytical solution of the master equation for the time evolution of the adatom site distribution, we study the diffusion properties of an isolated In adatom on InxGa1−xAs wetting layers (WL) deposited on the GaAs(001). The WL reconstructions considered in this study are, listed in the order of increasing In coverage: c(4 × 4), (1 × 3), (2 × 3), α2(2 × 4) and β2(2 × 4). We analyze the dependence of the diffusion properties on WL reconstruction, composition and strain, and find that: (i) diffusion on the (2 × N) reconstructions is strongly anisotropic, owing to the presence of the low barrier potential in-dimer trench, favoring the diffusion along the direction over that along the [110] direction; (ii) In diffusion at a WL coverage θ = 2/3 monolayers (ML; with composition x = 2/3) is faster than on clean GaAs(001) c(4 × 4), and decreases at θ = 1.75 ML (x = 1; e.g. InAs/GaAs(001)); (iii) diffusion and nucleation on the (2 × 4) WL is affected by the presence of adsorption sites for indium inside the As dimers; (iv) the approximation used for the exchange–correlation potential within DFT has an important effect on the description of the diffusion properties.',
        'Mongolian speech corpus for text-to-speech development [SEP] This paper presents a first attempt to develop Mongolian speech corpus that designed for data-driven speech synthesis in Mongolia. The aim of the speech corpus is to develop a high-quality Mongolian TTS for blinds to use with screen reader. The speech corpus contains nearly 6 hours of Mongolian phones. It well provides Cyrillic text transcription and its phonetic transcription with stress marking. It also provides context information including phone context, stressing levels, syntactic position in word, phrase and utterance for modeling speech acoustics and characteristics for speech synthesis.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Classification

* Dataset: `scidocs_cite_eval`
* Evaluated with [<code>CrossEncoderClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator)

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.8223     |
| accuracy_threshold    | 0.5698     |
| f1                    | 0.8341     |
| f1_threshold          | 0.4105     |
| precision             | 0.7777     |
| recall                | 0.8992     |
| **average_precision** | **0.8799** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### scirepeval

* Dataset: [scirepeval](https://huggingface.co/datasets/allenai/scirepeval) at [781d35d](https://huggingface.co/datasets/allenai/scirepeval/tree/781d35d1bf87253b3dcd0fadcb82bfbee9c244f1)
* Size: 6,170,503 training samples
* Columns: <code>query</code>, <code>pos</code>, and <code>neg</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                                              | pos                                                                                                 | neg                                                                                                 |
  |:--------|:---------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|
  | type    | string                                                                                             | string                                                                                              | string                                                                                              |
  | details | <ul><li>min: 47 characters</li><li>mean: 1274.16 characters</li><li>max: 9690 characters</li></ul> | <ul><li>min: 44 characters</li><li>mean: 1283.36 characters</li><li>max: 10088 characters</li></ul> | <ul><li>min: 37 characters</li><li>mean: 1225.97 characters</li><li>max: 10129 characters</li></ul> |
* Samples:
  | query                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | pos                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | neg                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Detecting tax evasion: a co-evolutionary approach [SEP] We present an algorithm that can anticipate tax evasion by modeling the co-evolution of tax schemes with auditing policies. Malicious tax non-compliance, or evasion, accounts for billions of lost revenue each year. Unfortunately when tax administrators change the tax laws or auditing procedures to eliminate known fraudulent schemes another potentially more profitable scheme takes it place. Modeling both the tax schemes and auditing policies within a single framework can therefore provide major advantages. In particular we can explore the likely forms of tax schemes in response to changes in audit policies. This can serve as an early warning system to help focus enforcement efforts. In addition, the audit policies can be fine tuned to help improve tax scheme detection. We demonstrate our approach using the iBOB tax scheme and show it can capture the co-evolution between tax evasion and audit policy. Our experiments shows the expect...</code> | <code>An Intelligent Anti-Money Laundering Method for Detecting Risky Users in the Banking Systems [SEP] During the last decades, universal economy has experienced money laundering and its destructive impact on the economy of the countries. Money laundering is the process of converting or transferring an asset in order to conceal its illegal source or assist someone that is involved in such crimes. Criminals generally attempt to clean the sources of the funds obtained by crime, using the banking system. Due to the large amount of information in the banks, detecting such behaviors is not feasible without anti-money laundering systems. Money laundering detection is one of the areas, where data mining tools can be useful and effective. In this research, some of the features of the users are extracted from their profiles by studying them. These features may include large financial transactions in risky areas regarding money laundering, reactivation of dormant accounts with considerable amounts, ...</code> | <code>Adapting nlp and corpus analysis techniques to structured imagery analysis in classical chinese poetry [SEP] This paper describes some pioneering work as a joint research project between City University of Hong Kong and Yuan Ze University in Taiwan to adapt language resources and technologies in order to set up a computational framework for the study of the creative language employed in classical Chinese poetry. ::: ::: In particular, it will first of all describe an existing ontology of imageries found in poems written during the Tang and the Song dynasties (7th -14th century AD). It will then propose the augmentation of such imageries into primary, complex, extended and textual imageries. A rationale of such a structured approach is that while poets may use a common dichotomy of primary imageries, creative language use is to be found in the creation of complex and compound imageries. This approach will not only support analysis of inter-poets stylistic similarities and differences bu...</code> |
  | <code>Lethal effects of abamectin on the aquatic organisms Daphnia similis, Chironomus xanthus and Danio rerio. [SEP] Abamectin is used as an acaricide and insecticide for fruits, vegetables and ornamental plants, as well as a parasiticide for animals. One of the major problems of applying pesticides to crops is the likelihood of contaminating aquatic ecosystems by drift or runoff. Therefore, toxicity tests in the laboratory are important tools to predict the effects of chemical substances in aquatic ecosystems. The aim of this study was to assess the potential hazards of abamectin to the freshwater biota and consequently the possible losses of ecological services in contaminated water bodies. For this purpose, we identified the toxicity of abamectin on daphnids, insects and fish. Abamectin was highly toxic, with an EC(50) 48 h for Daphnia similis of 5.1 ng L(-1), LC(50) 96 h for Chironomus xanthus of 2.67 μg L(-1) and LC(50) 48 h for Danio rerio of 33 μg L(-1).</code>                                  | <code>Effects of in vitro and in vivo avermectin exposure on alpha synuclein expression and proteasomal activity in pigeons. [SEP] Avermectins (AVMs) are used worldwide in agriculture and veterinary medicine. Residues of avermectin drugs, causing toxicological effects on non-target organisms, have raised great concern. The aim of this study was to investigate the effects of AVM on the expression levels of alpha synuclein (α-Syn) and proteasomal activity in pigeon (Columba livia) neurons both in vivo and in vitro. The results showed that, the mRNA and protein levels of α-Syn increased in AVM treated groups relative to control groups in the cerebrum, cerebellum and optic lobe in vivo. Dose-dependent decreases in the proteasomal activity (i.e., chymotrypsin-like, trypsin-like and peptidylglutamyl peptidehydrolase) were observed both in vivo and in vitro. The results suggested that AVM could induce the expression levels of α-Syn and inhibit the normal physiological function of proteasome in brai...</code> | <code>Impact of metabolic pathways and salinity on the hydrogen isotope ratios of haptophyte lipids [SEP] Abstract. Hydrogen isotope ratios of biomarkers have been shown to reflect water isotope ratios, and in some cases correlate significantly with salinity. The δ2H-salinity relationship is best studied for long-chain alkenones, biomarkers for haptophyte algae, and is known to be influenced by a number of different environmental parameters. It is not fully known why δ2H ratios of lipids retain a correlation to salinity, and whether this is a general feature for other lipids produced by haptophyte algae. Here, we analyzed δ2H ratios of three fatty acids, brassicasterol, long-chain C37 alkenones and phytol from three different haptophyte species grown over a range of salinities. Lipids synthesized in the cytosol, or relying on precursors of cytosolic origin, show a correlation between their δ2H ratios and salinity. In contrast, biosynthesis in the chloroplast, or utilizing precursors created ...</code> |
  | <code>IGF1 activates cell cycle arrest following irradiation by reducing binding of ΔNp63 to the p21 promoter [SEP] Radiotherapy for head and neck tumors often results in persistent loss of function in salivary glands. Patients suffering from impaired salivary function frequently terminate treatment prematurely because of reduced quality of life caused by malnutrition and other debilitating side-effects. It has been previously shown in mice expressing a constitutively active form of Akt (myr-Akt1), or in mice pretreated with IGF1, apoptosis is suppressed, which correlates with maintained salivary gland function measured by stimulated salivary flow. Induction of cell cycle arrest may be important for this protection by allowing cells time for DNA repair. We have observed increased accumulation of cells in G2/M at acute time-points after irradiation in parotid glands of mice receiving pretreatment with IGF1. As p21, a transcriptional target of the p53 family, is necessary for maintaining G2/M ...</code> | <code>ΔNp63 expression in four carcinoma cell lines and the effect on radioresistance—a siRNA knockdown model [SEP] ObjectivesThis study investigated the expression of ΔNp63α in carcinoma cell lines of the upper aerodigestive tract and their potential influence on radioresistance using a small interfering RNA (siRNA) knockdown approach.Materials and methodsFour carcinoma cell lines were investigated for the expression of the ΔNp63 isoform by quantitative reverse transcriptase polymerase chain reaction (qRT-PCR) (0, 24, 48 h) with and without single dose irradiation of 6 Gy. Furthermore, all cell lines were transfected with siRNA against the ΔNp63α isoform over 24 h. Knockdown effectiveness was controlled by qRT-PCR and Western blot. Apoptotic events were evaluated by terminal transferase dUTP nick end labeling (TUNEL) assay and cross-checked by a test for cell viability (WST-1, Roche) over 48 h.ResultsAll cell lines presented varying expression of the ΔNp63α isoform with and without irradiat...</code> | <code>Field preference of Greylag geese Anser anser during the breeding season [SEP] Few studies address food preference of geese on agricultural land (utilization related to availability) and only a handful so for the breeding season. We studied Greylag geese Anser anser during the breeding season in an intensively farmed area in southern Sweden. Few of 22 available field types were truly preferred. Pastureland was the most consistently preferred, by goslings (with parents) as well as by non-breeders. In some sampling periods, goslings also preferred grazed hay, ley, and carrot fields. Non-breeders exploited a greater variety of crops/fields, feeding also on barley, fallow, grazed hay, lettuce, oats, potatoes, and carrots. Most of these crops were preferred on at least one sampling occasion, except for fallow, grazed hay, and wheat, which were always used less than expected from availability. GLMs revealed that goslings rested more than they fed and preferred shorter vegetation before highe...</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 10.0,
      "num_negatives": 4,
      "activation_fn": "torch.nn.modules.activation.Sigmoid"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 40
- `weight_decay`: 0.01
- `num_train_epochs`: 0.67
- `warmup_steps`: 500
- `bf16`: True
- `dataloader_num_workers`: 4
- `ignore_data_skip`: True
- `resume_from_checkpoint`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 40
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.01
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 0.67
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 500
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: True
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 4
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: True
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: True
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step   | Training Loss | scidocs_cite_eval_average_precision |
|:------:|:------:|:-------------:|:-----------------------------------:|
| 0.0003 | 102052 | 0.2698        | -                                   |
| 0.0007 | 102114 | 0.2475        | -                                   |
| 0.0008 | 102125 | -             | 0.8795                              |
| 0.0011 | 102176 | 0.2632        | -                                   |
| 0.0015 | 102238 | 0.2729        | -                                   |
| 0.0016 | 102250 | -             | 0.8796                              |
| 0.0019 | 102300 | 0.2438        | -                                   |
| 0.0023 | 102362 | 0.2364        | -                                   |
| 0.0024 | 102375 | -             | 0.8796                              |
| 0.0027 | 102424 | 0.2456        | -                                   |
| 0.0032 | 102486 | 0.2348        | -                                   |
| 0.0032 | 102500 | -             | 0.8796                              |
| 0.0036 | 102548 | 0.2394        | -                                   |
| 0.0040 | 102610 | 0.2467        | -                                   |
| 0.0041 | 102625 | -             | 0.8797                              |
| 0.0044 | 102672 | 0.2412        | -                                   |
| 0.0048 | 102734 | 0.2529        | -                                   |
| 0.0049 | 102750 | -             | 0.8798                              |
| 0.0052 | 102796 | 0.2388        | -                                   |
| 0.0056 | 102858 | 0.2666        | -                                   |
| 0.0057 | 102875 | -             | 0.8799                              |
| 0.0060 | 102920 | 0.2417        | -                                   |
| 0.0064 | 102982 | 0.2412        | -                                   |
| 0.0065 | 103000 | -             | 0.8799                              |
| 0.0068 | 103044 | 0.2439        | -                                   |
| 0.0072 | 103106 | 0.2466        | -                                   |
| 0.0073 | 103125 | -             | 0.8799                              |
| 0.0076 | 103168 | 0.2382        | -                                   |
| 0.0080 | 103230 | 0.2487        | -                                   |
| 0.0081 | 103250 | -             | 0.8799                              |
| 0.0084 | 103292 | 0.244         | -                                   |
| 0.0088 | 103354 | 0.248         | -                                   |


### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 5.2.0
- Transformers: 4.57.3
- PyTorch: 2.9.1+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.2
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->