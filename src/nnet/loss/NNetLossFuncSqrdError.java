package nnet.loss;

import nnet.matrix.Matrix;

public class NNetLossFuncSqrdError implements NNetLossFunction {

	public NNetLossFuncSqrdError() {
	}

	@Override
	public double getLoss(Matrix actual, Matrix expected) {
		return actual.subtract(expected).square().sum();
	}

}
