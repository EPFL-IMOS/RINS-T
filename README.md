# RINS-T

This paper has been officially accepted for publication in the **IEEE Transactions on Instrumentation and Measurement**.
___

**Abstract:** 

Time series data are often affected by various forms of corruption, such as missing values, noise, and outliers, which pose significant challenges for tasks such as forecasting and anomaly detection. To address these issues, inverse problems focus on reconstructing the original signal from corrupted data by leveraging prior knowledge about its underlying structure. While deep learning methods have demonstrated potential in this domain, they often require extensive pretraining and struggle to generalize under distribution shifts. In this work, we propose RINS-T (Robust Implicit Neural Solvers for Time Series Linear Inverse Problems), a novel deep prior framework that achieves high recovery performance without requiring pretraining data. RINS-T leverages neural networks as implicit priors and integrates robust optimization techniques, making it resilient to outliers while relaxing the reliance on Gaussian noise assumptions. To further improve optimization stability and robustness, we introduce three key innovations: guided input initialization, input perturbation, and convex output combination techniques. Each of these contributions strengthens the framework's optimization stability and robustness. These advancements make RINS-T a flexible and effective solution for addressing complex real-world time series challenges.

___ 


