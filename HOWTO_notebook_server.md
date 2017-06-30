Should you want to use jupyter notebooks to communicate and code on the GPU sever, you can do as follows.

Think of a port number that noone else probably uses, something random like 8213. Then, connect to the server with SSH like:

    ssh -L 8213:localhost:8213 student_X@hgsgpu01.iwr.uni-heidelberg.de

Connect via SSH. Type screen first, so we don't stop our server if we disconnect:

    screen
  
I recommend to then create a virtual environment and activate it:

    python3 -m venv env
    . env/bin/activate
  
A virtual environment isolates your Python dependencies from the rest of the system and therefore allows you to install Python packages without sudo rights.
The (env) in front of your shell prompt tells you that the virtual environment is active for the session.
Use it to install Jupyter notebook and everything else you need. We install a recent pip first to make it faster:

    pip install -U pip wheel
    pip install jupyter numpy …

Select the GPU you want:

    export CUDA_VISIBLE_DEVICES="4"

Then, run your notebook on your port:

    jupyter notebook --port=8213
  
Open the browser on your local computer and point it to the link outputted by the above command. Tada!

To "close" the screen session without stopping the notebook, press Ctrl+A, followed by D (not at the same time).
Then, close the shell.

To reconnect, just use the same SSH command line again. You just need to type anything, the session just needs to stay open.
If you want to interrupt the notebook, use ``screen -r`` to jump into your screen session and press Ctrl+C to end it.

Have fun!
