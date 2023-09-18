package nnet.matrix.net;

import nnet.matrix.NNetPerceptron;

public class TrainingStats {
	public int totalIterations;
	public double finalLoss;
	public NNetPerceptron finalResult; 

	public TrainingStats() {
		clear();
	}

	public void clear() {
		totalIterations = 0;
		finalLoss = 0;
		finalResult = null;
		
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("Training stats: iterations=");
		sb.append(totalIterations);
		sb.append(" finalLoss=");
		sb.append(String.format("%,.8f",finalLoss));
		sb.append("\n");
		sb.append(finalResult.toString("results"));

		return sb.toString();
	}

	
	
}
