# HiCur-NPC: Hierarchical Feature Fusion Curriculum Learning for Multi-Modal Foundation Model in Nasopharyngeal Carcinoma
<p align="center">
    <img src="images/cartoon.png" width="50%"> <br>
</p>
Providing precise and comprehensive diagnostic information to clinicians is crucial for improving the treatment and prognosis of nasopharyngeal carcinoma. Multi-modal foundation models, which can integrate data from various sources, have the potential to significantly enhance clinical assistance. However, several challenges remain:

1. The lack of large-scale visual-language datasets for nasopharyngeal carcinoma.
2. Existing pre-training and fine-tuning methods that cannot learn the necessary hierarchical features for complex clinical tasks.
3. Current foundation models having limited visual perception due to inadequate integration of multi-modal information.

While curriculum learning can improve a model's ability to handle multiple tasks through systematic knowledge accumulation, it still lacks consideration for hierarchical features and their dependencies, affecting knowledge gains. To address these issues, we propose the Hierarchical Feature Fusion Curriculum Learning (HFFCL) method, which consists of three stages:

1. **Visual Knowledge Learning (Stage I)**: We introduce the Hybrid Contrastive Masked Autoencoder (HCMAE) to pre-train visual encoders on 755K multi-modal images of nasopharyngeal carcinoma CT, MRI, and endoscopy to fully extract deep visual information.
2. **Coarse-Grained Alignment (Stage II)**: We construct a 65K visual instruction fine-tuning dataset based on open-source data and clinician diagnostic reports, achieving coarse-grained alignment with visual information in a large language model (CGA).
3. **Fine-Grained Fusion (Stage III)**: We design a Mixture of Experts Cross Attention (MECA) structure for deep fine-grained fusion of global multimodal information.

Our model outperforms previously developed specialized models in all key clinical tasks for nasopharyngeal carcinoma, including diagnosis, report generation, tumor segmentation, and prognosis.
![HiCur](images/HiCur.png)
## Repository Structure

- `StageI-HCMAE`: Contains code and resources for visual knowledge learning using the Hybrid Contrastive Masked Autoencoder.
- `StageII-CGA`: Includes scripts and datasets for coarse-grained alignment.
- `StageIII-FGF`: Hosts the implementation for fine-grained fusion using the Mixture of Experts Cross Attention structure.
- `test`: Provides the complete model architecture and inference examples.

## Note

This is not the full version of the repository. Some code is currently being refined and will be released once it has been validated and reconstructed to ensure usability.

## Installation

To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

Detailed instructions for each stage can be found within their respective folders. To test the complete model, navigate to the `test` directory and follow the instructions in the README file provided there.

## Contributing

We welcome contributions from the community. Please fork the repository and submit a pull request with your changes. Ensure your code adheres to our style guidelines and includes appropriate tests.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or inquiries, please contact us at wangzipei23@ia.ac.cn.
