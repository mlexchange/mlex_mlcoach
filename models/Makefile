TAG 			:= latest	
USER 			:= mlexchange
PROJECT			:= tensorflow-neural-networks

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

run_docker:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ -v /mnt/c/Users/postdoc/Documents/Database/born:/app/work/data -p 8888:8888 ${IMG_WEB_SVC}

train_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v /data/tanchavez/Datasets/born:/app/work/data ${IMG_WEB_SVC} python3 src/train_model.py data/train data/val data/logs '{"rotation_angle": 0, "image_flip": "None", "batch_size": 32, "pooling": "None", "epochs": 10, "nn_model": "ResNet50"}' 

evaluate_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v /data/tanchavez/Datasets/born:/app/work/data ${IMG_WEB_SVC} python3 src/evaluate_model.py data/test data/logs/model.h5 '{"rotation_angle": 0, "image_flip": "None", "batch_size": 1}'

predict_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v /data/tanchavez/Datasets/born:/app/work/data ${IMG_WEB_SVC} python3 src/predict_model.py data/test data/logs/model.h5 data/logs '{"rotation_angle": 0, "image_flip": "None", "batch_size": 1}'

transfer_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it --gpus all -v /data/tanchavez/Datasets/born:/app/work/data ${IMG_WEB_SVC} python3 src/transfer_learning.py data/train data/val data/logs/model.h5 data/logs '{"rotation_angle": 0, "image_flip": "None", "batch_size": 32, "pooling": "None", "epochs": 10, "init_layer": 100}'


clean: 
	find -name "*~" -delete
	-rm .python_history
	-rm -rf .config
	-rm -rf .cache

push_docker:
	docker push ${IMG_WEB_SVC}
