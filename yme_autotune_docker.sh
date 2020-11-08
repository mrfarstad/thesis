scp -r ./* yme:~/thesis_autotune
ssh yme -t "
    cd thesis_autotune;
    docker build . -t nvidia-test;
    "

    #docker create -it --gpus all --name laplace3d nvidia-test bash;
    #docker cp laplace3d:/usr/src/laplace3d/results/laplace3d.png .
