# mlex_mlcoach
## Description
This app provides a training/testing platform for image classification with supervised deep-learning approaches.

## Running as a standalone application

1. Start the compute and content services in the [MLExchange platform](https://github.com/mlexchange/mlex). Before moving to the next step, please make sure that the computing API and the content registry are up and running. For more information, please refer to their respective
README files.

2. Start [splash-ml](https://github.com/als-computing/splash-ml)

2. Create a new Python environment and install dependencies:
```
conda create -n new_env python==3.11
conda activate new_env
pip install .
```

3. Create a `.env` file using `.env.example` as reference. Update this file accordingly.

4. Start example app:
```
python frontend.py
```

Finally, you can access MLCoach at:
* Dash app: http://localhost:8062/

Please refer to [HowToGuide](/docs/tasks.md) for further instructions on how
to use this application.

# Model Description
**TF-NeuralNetworks:** Assortment of neural networks implemented in [TensorFlow](https://www.tensorflow.org).

Further information can be found in [mlex_image_classification](https://github.com/mlexchange/mlex_image_classification).

To make existing algorithms available in MLCoach, make sure to upload the `model description` to the content registry.

# Copyright
MLExchange Copyright (c) 2024, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
