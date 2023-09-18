package nnet.matrix.net;

import nnet.matrix.NNetPerceptron;

public interface NetworkListener {
	public void shouldContinuer(int iteration,double loss,NNetPerceptron predicition);

}
