TAG 			:= latest	
USER 			:= mlexchange
PROJECT			:= labelmaker-functions

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
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}/models:/app/work ${IMG_WEB_SVC} python3 src/train_model.py '{"rotation_angle": 0, "image_flip": [], "batch_size": 64, "pooling": "None", "stepoch": 1, "epochs": 1, "nn_model": "VGG16"}'

test_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ ${IMG_WEB_SVC} python3 scr/segment.py data/images/test/segment_series.tif data/output/state_dict_net.pt data/output '{"show_progress": 20}'


clean: 
	find -name "*~" -delete
	-rm .python_history
	-rm -rf .config
	-rm -rf .cache

push_docker:
	docker push ${IMG_WEB_SVC}