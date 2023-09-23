package nnet.matrix.net;

import java.util.ArrayList;
import java.util.List;

import nnet.matrix.NNetMatrix;

public class TrainingStats {
	public int totalIterations;
	public double finalLoss;
	public NNetMatrix finalResult; 
	public List<Double> lossArray;

	public TrainingStats() {
		clear();
	}

	public void clear() {
		totalIterations = 0;
		finalLoss = 0;
		finalResult = null;
		lossArray = new ArrayList<Double>();
	}
	
	public void addLossResult(double loss) {
		lossArray.add(loss);
	}

	public List<Double> getLossArray() {
		return lossArray;
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
