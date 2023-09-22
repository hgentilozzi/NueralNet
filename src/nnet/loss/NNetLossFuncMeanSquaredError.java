package nnet.loss;

import nnet.matrix.Matrix;

public class NNetLossFuncMeanSquaredError implements NNetLossFunction {

	public NNetLossFuncMeanSquaredError() {
	}

	@Override
	public double getLoss(Matrix actual, Matrix expected) {
		double sumsqrs = actual.subtract(expected).square().sum();
		return (sumsqrs/expected.getCols());
	}

}
