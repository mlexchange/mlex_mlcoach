# mlex_mlcoach
## Description
This app provides a training/testing platform for image classification with supervised deep-learning approaches.

## Running as a standalone application
First, let's install docker:

* https://docs.docker.com/engine/install/

Next, let's [install the MLExchange platform](https://github.com/mlexchange/mlex).

Before moving to the next step, please make sure that the computing API and the content 
registry are up and running. For more information, please refer to their respective 
README files.

* Clone this repository in your local device.
* Next, cd into mlex_mlcoach
* type `docker-compose up` into your terminal

Finally, you can access MLCoach at:
* Dash app: http://localhost:8062/

Please refer to [HowToGuide](/docs/tasks.md) for further instructions on how
to use this application.

# Model Description
**TF-NeuralNetworks:** Assortment of neural networks implemented in [TensorFlow](https://www.tensorflow.org).

Further information can be found in [concepts](/docs/concepts.md).

# Copyright
MLExchange Copyright (c) 2023, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
