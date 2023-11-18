#! /bin/bash

first_session=$1

nohup python -u ccuhub.py > step1.log 2>&1 &
s1_pid=$!
echo "Step 1 PID: $s1_pid"
nohup python -u audio_proxy.py > step4.log 2>&1 &
s4_pid=$!
echo "Step 4 PID: $s4_pid"
nohup python -u webcam_proxy.py > step5.log 2>&1 &
s5_pid=$!
echo "Step 5 PID: $s5_pid"
nohup python -u asr.py > step6.log 2>&1 &
s6_pid=$!
echo "Step 6 PID: $s6_pid"
nohup python -u monash_norm_detector.py > step7.log 2>&1 &
s7_pid=$!
echo "Step 7 PID: $s7_pid"

docker load -i parc-purdue-text-service.tar.gz
docker run -d --name parc-purdue -it --rm --publish-all --volume $CCU_SANDBOX:/sandbox parc-purdue-text-service > step8.log 2>&1 &
# echo "Step 8 PID: $!"

docker load -i monash-subteam3-update.tar.gz
docker run -d --name subteam3_cp -it --rm --publish-all --volume $CCU_SANDBOX:/sandbox subteam3_cp > step10.log 2>&1 &
# echo "Step 10 PID: $!"

# docker load -i monash-ta2-rewrite.tar
# docker run -d -it --rm --publish-all --volume $CCU_SANDBOX:/sandbox monash-ta2-rewrite > rewrite.log
# echo "Step 9 PID: $!"


for (( ; ; ))
do
	read -p "Enter The Name of a New Session / Scenario: " first_session
	docker run -d --name monash-ta2-rewrite -it --rm --publish-all --volume $CCU_SANDBOX:/sandbox monash-ta2-rewrite
	# docker run -d -it --rm --publish-all --volume $CCU_SANDBOX:/sandbox monash-ta2-rewrite 
	nohup python -u logger.py --jsonl ${first_session}.jsonl > step2.log 2>&1 &
	logger_pid=$!
	echo "Step 2 PID: $logger_pid"
	nohup python -u message_proxy.py > step3.log 2>&1 &
	message_pid=$!
	echo "Step 3 PID: $message_pid"

	python -u ta2_gpt3.py

	kill -9 $logger_pid $message_pid
	docker_id=$(docker ps -aqf "name=monash-ta2-rewrite")
	docker kill $docker_id

	printf "=========================\nCurrent Session / Scenario ended.\n The Log is saved at $(pwd)/${first_session}.jsonl\n=========================\n"

	read -p "Do you want to start a New Session / Scenario (Y/N)? 'Y' means start a new session and 'N' refers to terminating all the processes." continue_flag

	if [ $continue_flag == "N" ]
	then
		break
	fi

done

docker_id=$(docker ps -aqf "name=parc-purdue")
docker kill $docker_id

docker_id=$(docker ps -aqf "name=subteam3_cp")
docker kill $docker_id

kill $s1_pid $s4_pid $s5_pid $s6_pid $s7_pid

echo "Thank you for Using Monash TA2 System!"






