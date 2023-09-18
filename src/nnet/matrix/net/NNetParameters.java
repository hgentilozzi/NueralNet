package nnet.matrix.net;

public class NNetParameters {
	private static NNetParameters instance = null;
	
	private double learningRate = 0.0001;
	
	public static NNetParameters getInstance() {
		if (instance==null)
			instance = new NNetParameters();
		return instance;
	}

	public double getLearningRate() {
		return learningRate;
	}

	

}
