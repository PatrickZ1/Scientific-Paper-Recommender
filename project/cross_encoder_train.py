from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset

from recomm_dataset import load_scidocs_cite, scidoc_cite_to_triplets

model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", max_length=512)

# TODO: use more than 1/100 of the training data for quick testing
train_dataset = scidoc_cite_to_triplets(load_scidocs_cite()["train"].shard(100, 0))

# TODO: Hard Example Mining or Loss Adjustment -> almost all negatives are too easy (will give a high score for paper from roughly the same field)

loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=256)
args = CrossEncoderTrainingArguments(
    output_dir="./cross-encoder-checkpoints",
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    num_train_epochs=1,
    resume_from_checkpoint=True,
    per_device_train_batch_size=256,
)
trainer = CrossEncoderTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
    args=args,
)
trainer.train(resume_from_checkpoint=True)

query = (
    '"Surface Reconstruction from Point Clouds via Grid-based Intersection Prediction"\n'
    "Surface reconstruction from point clouds is a crucial task in the fields of computer vision and computer graphics. SDF-based methods excel at reconstructing smooth meshes with minimal error and artefacts but struggle with representing open surfaces. On the other hand, UDF-based methods can effectively represent open surfaces but often introduce noise, leading to artefacts in the mesh. In this work, we propose a novel approach that directly predicts the intersection points between line segment of point pairs and implicit surfaces. To achieve it, we propose two modules named Relative Intersection Module and Sign Module respectively with the feature of point pair as input. To preserve the continuity of the surface, we also integrate symmetry into the two modules, which means the position of predicted intersection will not change even if the input order of the point pair changes. This method not only preserves the ability to represent open surfaces but also eliminates most artefacts on the mesh. Our approach demonstrates state-of-the-art performance on three datasets: ShapeNet, MGN, and ScanNet. The code will be made available upon acceptance."
)

passages = [
    '"Neural Point-Based Graphics"\nWe present a neural point-based graphics (NPBG) pipeline for real-time rendering of scenes using point clouds as the underlying representation. NPBG combines ideas from traditional point-based graphics with recent advances in neural rendering to generate high-quality images at real-time frame rates. Our approach uses a neural network to learn view-dependent appearance effects, such as specular highlights and reflections, directly from the point cloud data. We demonstrate the effectiveness of our method on a variety of scenes, showing that NPBG can produce visually compelling results while maintaining interactive performance.',
    '"Learning Implicit Fields for Generative Shape Modeling"\nWe propose a novel approach for generative shape modeling using implicit fields. Our method leverages deep neural networks to learn continuous implicit representations of 3D shapes, allowing for high-quality shape generation and interpolation. We introduce a new architecture that combines convolutional layers with implicit function representations, enabling the model to capture complex geometric details. We evaluate our approach on several benchmark datasets, demonstrating its ability to generate diverse and realistic 3D shapes while outperforming existing methods in terms of quality and fidelity.',
    '"CAD-NeRF: learning NeRFs from uncalibrated few-view images by CAD model retrieval"\nReconstructing from multi-view images is a longstanding problem in 3D vision, where neural radiance fields (NeRFs) have shown great potential and get realistic rendered images of novel views. Currently, most NeRF methods either require accurate camera poses or a large number of input images, or even both. Reconstructing NeRF from few-view images without poses is challenging and highly ill-posed. To address this problem, we propose CAD-NeRF, a method reconstructed from less than 10 images without any known poses. Specifically, we build a mini library of several CAD models from ShapeNet and render them from many random views. Given sparse-view input images, we run a model and pose retrieval from the library, to get a model with similar shapes, serving as the density supervision and pose initializations. Here we propose a multi-view pose retrieval method to avoid pose conflicts among views, which is a new and unseen problem in uncalibrated NeRF methods. Then, the geometry of the object is trained by the CAD guidance. The deformation of the density field and camera poses are optimized jointly. Then texture and density are trained and fine-tuned as well. All training phases are in self-supervised manners. Comprehensive evaluations of synthetic and real images show that CAD-NeRF successfully learns accurate densities with a large deformation from retrieved CAD models, showing the generalization abilities.',
    '"S2M-Net: Spectral-Spatial Mixing for Medical Image Segmentation with Morphology-Aware Adaptive Loss"\nMedical image segmentation requires balancing local precision for boundary-critical clinical applications, global context for anatomical coherence, and computational efficiency for deployment on limited data and hardware a trilemma that existing architectures fail to resolve. Although convolutional networks provide local precision at $mathcal{O}(n)$ cost but limited receptive fields, vision transformers achieve global context through $mathcal{O}(n^2)$ self-attention at prohibitive computational expense, causing overfitting on small clinical datasets. We propose S2M-Net, a 4.7M-parameter architecture that achieves $mathcal{O}(HW log HW)$ global context through two synergistic innovations: (i) Spectral-Selective Token Mixer (SSTM), which exploits the spectral concentration of medical images via truncated 2D FFT with learnable frequency filtering and content-gated spatial projection, avoiding quadratic attention cost while maintaining global receptive fields; and (ii) Morphology-Aware Adaptive Segmentation Loss (MASL), which automatically analyzes structure characteristics (compactness, tubularity, irregularity, scale) to modulate five complementary loss components through constrained learnable weights, eliminating manual per-dataset tuning. Comprehensive evaluation in 16 medical imaging datasets that span 8 modalities demonstrates state-of-the-art performance: 96.12% Dice on polyp segmentation, 83.77% on surgical instruments (+17.85% over the prior art) and 80.90% on brain tumors, with consistent 3-18% improvements over specialized baselines while using 3.5--6$times$ fewer parameters than transformer-based methods.',
    '"Gradient Boosting Trees and Large Language Models for Tabular Data Few-Shot Learning"\nLarge Language Models (LLM) have brought numerous of new applications to Machine Learning (ML). In the context of tabular data (TD), recent studies show that TabLLM is a very powerful mechanism for few-shot-learning (FSL) applications, even if gradient boosting decisions trees (GBDT) have historically dominated the TD field. In this work we demonstrate that although LLMs are a viable alternative, the evidence suggests that baselines used to gauge performance can be improved. We replicated public benchmarks and our methodology improves LightGBM by 290%, this is mainly driven by forcing node splitting with few samples, a critical step in FSL with GBDT. Our results show an advantage to TabLLM for 8 or fewer shots, but as the number of samples increases GBDT provides competitive performance at a fraction of runtime. For other real-life applications with vast number of samples, we found FSL still useful to improve model diversity, and when combined with ExtraTrees it provides strong resilience to overfitting, our proposal was validated in a ML competition setting ranking first place.',
]

ranks = model.rank(query, passages)

print("Query:", query)
for rank in ranks:
    print(f"{rank['score']:.2f}\t{passages[rank['corpus_id']]}")
