package nnet.loss;

import nnet.matrix.Matrix;

public class NNetLossFuncAbsoluteError implements NNetLossFunction {

	public NNetLossFuncAbsoluteError() {
	}

	@Override
	public double getLoss(Matrix actual, Matrix expected) {
		double sumsqrs = actual.subtract(expected).absoluteValue().sum();
		return (sumsqrs/expected.getCols());
	}

}
