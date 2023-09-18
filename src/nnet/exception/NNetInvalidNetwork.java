package nnet.exception;

/**
 * Internal training error
 */
public class NNetInvalidNetwork extends Exception {
	private static final long serialVersionUID = 4966977196249006324L;

	public NNetInvalidNetwork() {
	}

	public NNetInvalidNetwork(String message) {
		super(message);
	}
}
