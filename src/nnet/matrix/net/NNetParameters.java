package nnet.matrix.net;

public class NNetParameters {
	private static NNetParameters instance = null;
	
	private double learningRate = 0.0001;
	private int batchSize = 1;
	private boolean useSoftMax = false;
	
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

	public int getBatchSize() {
		return batchSize;
	}

	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}

	public boolean isUseSoftMax() {
		return useSoftMax;
	}

	public void setUseSoftMax(boolean useSoftMax) {
		this.useSoftMax = useSoftMax;
	}


	

}
