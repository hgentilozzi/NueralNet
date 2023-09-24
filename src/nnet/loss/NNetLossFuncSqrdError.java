package nnet.loss;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.matrix.NNetMatrix;

public class NNetLossFuncSqrdError implements NNetLossFunction {

	public NNetLossFuncSqrdError() {
	}

	@Override
	public double getLoss(NNetMatrix actual, NNetMatrix expected) throws MxlibInvalidMatrixOp {
		return actual.subtract(expected).squareElement().sum();
	}

}
