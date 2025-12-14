# F-CTVAE: Federated Constrained Twin Variational Auto-Encoder for Privacy-Preserving Intrusion Detection in IoT Networks

**F-CTVAE** is a privacy-preserving, semi-supervised framework that federates the state-of-the-art **Constrained Twin Variational Auto-Encoder (CTVAE)** using the **Flower (flwr)** framework for intrusion detection in highly heterogeneous IoT networks.

### Key Features
- **Full raw data privacy**: No raw data ever leaves client devices.
- **62% reduction in communication overhead**: Only the hermaphrodite mapper, decoder, and class-specific Gaussian priors are selectively aggregated; local encoders remain strictly private.
- **Custom Constrained Momentum Gaussian Aggregation (CMGA) strategy**:
  - Momentum-enhanced updates for class-conditioned priors
  - Proximal enforcement of inter-class separation in latent space
  - Preserves (and enhances) discriminative structure under severe non-IID conditions
- **Superior performance over centralized CTVAE**:
  - Average macro F1-score: **96.70%** (vs. 92.88% for centralized CTVAE, **+3.82 pp improvement**)
  - Peak macro F1-score up to **99.37%**
  - Accuracy up to **98.75%**

### Dataset
Evaluated on the benchmark **N-BaIoT** dataset with six highly non-IID device-specific clients simulating real-world IoT botnet traffic (Mirai and Bashlite attacks).

### Implementation
- Built with **PyTorch 2.1** and **Flower 1.8**
- Custom Flower Strategy for selective and constrained aggregation
- Includes server pre-training, client fine-tuning, and real-time inference algorithms

### Results
F-CTVAE consistently outperforms the centralized CTVAE baseline in macro F1-score across all clients while drastically reducing bandwidth usage and fully preserving privacy.

For details, see the paper:  
**F-CTVAE: A Federated Constrained Twin VAE for Privacy-Preserving Intrusion Detection in IoT Networks**  
Ali Mousavi, Somayeh Changiz, Ehsan Baghshani, Mohammad Ansarimehr

### Repository Contents
- Source code for model architecture, federated training, and evaluation
- Scripts for reproducing experiments on the N-BaIoT dataset
- Configuration files and pre-trained models

### Citation
If you use this code or method in your research, please cite:

```bibtex
@article{mousavi2025fctvae,
  title={F-CTVAE: A Federated Constrained Twin VAE for Privacy-Preserving Intrusion Detection in IoT Networks},
  author={Mousavi, Ali and Changiz, Somayeh and Baghshani, Ehsan and Ansarimehr, Mohammad},
  journal={IEEE Transactions on ...}, % Update with actual publication details if available
  year={2025}
}