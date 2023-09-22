package nnet.matrix.net;

import nnet.matrix.acvt.ActivationFunction;

public class NNetParameters {
	private static NNetParameters instance = null;
	
	private double learningRate = 0.0001;
	private ActivationFunction activationFunction = ActivationFunction.LINEAR;
	private int batchSize = 1;
	
	public static NNetParameters getInstance() {
		if (instance==null)
			instance = new NNetParameters();
		return instance;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	public void setActivationFunction(ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}


	

}
