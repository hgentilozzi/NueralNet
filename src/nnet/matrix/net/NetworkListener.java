package nnet.matrix.net;

import nnet.matrix.NNetMatrix;

public interface NetworkListener {
	public void shouldContinuer(int iteration,double loss,NNetMatrix predicition);

}
