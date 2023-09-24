package nnet.matrix.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import nnet.matrix.NNetMatrix;

/**
 * A batch based on a set of arrays
 */
public class NNetBatchDataFile implements NnetBatchDataIntf {
	private int numInNodes;
	private int numOutNodes;
	private String error;
	private BufferedReader reader;
	private String fileName;
	private String nextLine;

	public NNetBatchDataFile(String fileName,int numInNodes,int numOutNodes)  {
		this.error = "";
		this.fileName = fileName;
		this.reader = null;
		this.nextLine = null;
		this.numInNodes = numInNodes;
		this.numOutNodes = numOutNodes;
	
		try {
			reset();
		} catch (Exception e) {
			closeReader();
			error="NNetBatchDataFile: Failed to initialize. Error=" + e.getMessage();
		}
			
	}

	private void closeReader() {
		if (reader!=null)
			try {
				reader.close();
			} catch (IOException e) {
			}
		reader = null;
	}
	
	/**
	 * Read the next line from the file. Skip comment lines
	 * @return - next line from file
	 * @throws Exception
	 */
	private String nextLine() throws Exception {
		String nl=null;
		while ((nl=reader.readLine())!=null && (nl.isBlank() || nl.startsWith("#")));
		return nl;
	}
	
	@Override
	public void reset() throws Exception {
		// Close the current file if already open
		closeReader();
		
		// Open the file
		reader = new BufferedReader(new FileReader(fileName));
		
		// pre-read the first line of data
		nextLine = nextLine();
	}


	@Override
	public boolean atEof() {
		return nextLine==null;
	}

	@Override
	public NNetBatch nextBatch(int batchSize) throws Exception {
		NNetBatch ret = null;
		List<String> batchlines = new ArrayList<String>();
		
		if (atEof())
			return null;
		
		batchlines.add(nextLine);
		int i=1;
		String l;
		// Read up to batch size of rows
		while (i++<batchSize && (l=nextLine())!=null){
			batchlines.add(l);
		}
		
		// buffer the next line for eof checking
		if (!atEof())
			nextLine = nextLine();
		
		// create the NNetMatrix for input and output
		double[][] indate = new double[batchlines.size()][numInNodes];
		double[][] outdate = new double[batchlines.size()][numOutNodes];
		int row = 0;

		for (String bstr : batchlines) {
			String[] parts = bstr.split(",");
			if (parts.length!=(numInNodes+numOutNodes)) {
				closeReader();
				reader = null;
				error = "NNetBatchDataFile: The data line is ill-formed: line=" + bstr;
				throw new Exception(error);
			}
			
			for (int col=0;col<numInNodes;col++)
				indate[row][col] = Double.parseDouble(parts[col]);

			for (int col=0;col<numOutNodes;col++)
				outdate[row][col] = Double.parseDouble(parts[numInNodes+col]);

			row++;
			
		}

		ret = new NNetBatch(batchlines.size(),new NNetMatrix(indate), new NNetMatrix(outdate));

		return ret;
	}

	@Override
	public boolean validate() {
		// Reader failed to open Error already set
		if (reader==null)
			return false;
		
		return true;
	}

	@Override
	public String getError() {
		return error;
	}

	public int getNumInNodes() {
		return numInNodes;
	}

	public int getNumOutNodes() {
		return numOutNodes;
	}


}
