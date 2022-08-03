TAG 			:= latest	
USER 			:= mlexchange1
PROJECT			:= tensorflow-neural-networks
DATA_PATH               := /data/tanchavez/Datasets/born
DATA_PATH_NPZ           := /data/tanchavez/Datasets

IMG_WEB_SVC    		:= ${USER}/${PROJECT}:${TAG}
IMG_WEB_SVC_JYP    	:= ${USER}/${PROJECT_JYP}:${TAG}
ID_USER			:= ${shell id -u}
ID_GROUP			:= ${shell id -g}

.PHONY:

test:
	echo ${IMG_WEB_SVC}
	echo ${TAG}
	echo ${PROJECT}
	echo ${PROJECT}:${TAG}
	echo ${ID_USER}

build_docker: 
	docker build -t ${IMG_WEB_SVC} -f ./docker/Dockerfile .

build_docker_gpu: 
	docker build -t ${IMG_WEB_SVC} -f ./docker/Dockerfile_gpu .

build_docker_arm64: 
	docker build -t ${IMG_WEB_SVC} -f ./docker/Dockerfile_arm64 .

run_docker:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ -v ${DATA_PATH}:/app/work/data -p 8888:8888 ${IMG_WEB_SVC}

train_example_gpu:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v ${DATA_PATH}:/app/work/data ${IMG_WEB_SVC} python3 src/train_model.py data/train data/logs '{"rotation_angle": 0, "image_flip": "None", "batch_size": 32, "val_pct": 0, "shuffle": true, "seed": 45, "pooling": "None", "optimizer": "Adam", "loss_function": "categorical_crossentropy", "learning_rate": 0.01, "epochs": 3, "nn_model": "ResNet50"}' 

train_example_cpu:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${DATA_PATH}:/app/work/data ${IMG_WEB_SVC} python3 src/train_model.py data/train data/logs '{"rotation_angle": 0, "image_flip": "None", "batch_size": 32, "val_pct": 10, "shuffle": true, "seed": 45, "pooling": "None", "optimizer": "Adam", "loss_function": "categorical_crossentropy", "learning_rate": 0.01, "epochs": 3, "nn_model": "ResNet50"}' 

train_example_npz:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v ${DATA_PATH_NPZ}:/app/work/data ${IMG_WEB_SVC} python3 src/train_model.py data/test.npz data/logs '{"rotation_angle": 0, "image_flip": "None", "batch_size": 32, "val_pct": 10, "shuffle": true, "seed": 45, "pooling": "None", "optimizer": "Adam", "loss_function": "categorical_crossentropy", "learning_rate": 0.01, "epochs": 3, "nn_model": "ResNet50", "x_key": "x_train", "y_key": "y_train"}'

evaluate_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${DATA_PATH}:/app/work/data ${IMG_WEB_SVC} python3 src/evaluate_model.py data/test data/logs/model.h5 '{"rotation_angle": 0, "image_flip": "None", "batch_size": 1}'

predict_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${DATA_PATH}:/app/work/data ${IMG_WEB_SVC} python3 src/predict_model.py data/test data/logs/model.h5 data/logs '{"rotation_angle": 0, "image_flip": "None", "batch_size": 32}'

transfer_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v ${DATA_PATH}:/app/work/data ${IMG_WEB_SVC} python3 src/transfer_learning.py data/train data/logs/model.h5 data/logs '{"rotation_angle": 0, "image_flip": "None", "batch_size": 32, "val_pct": 20, "shuffle": true, "seed": 45, "pooling": "None","optimizer": "Adam", "loss_function": "categorical_crossentropy", "learning_rate": 0.01, "epochs": 3, "init_layer": 100}'


clean: 
	find -name "*~" -delete
	-rm .python_history
	-rm -rf .config
	-rm -rf .cache

push_docker:
	docker push ${IMG_WEB_SVC}
