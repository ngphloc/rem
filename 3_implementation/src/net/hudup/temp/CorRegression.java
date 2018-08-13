package net.hudup.temp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.NextUpdate;
import net.hudup.core.parser.TextParserUtil;
import net.hudup.regression.AbstractRegression;

/**
 * This class implements regression model with correlation in case of missing data, called COR algorithm. 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@NextUpdate
public class CorRegression extends AbstractRegression implements DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of X vectors
	 */
	protected List<List<Double>> xVectors = Util.newList();
	
	
	/**
	 * Z vector.
	 */
	protected List<Double> zVector = Util.newList();

	
	/**
	 * Default constructor.
	 */
	public CorRegression() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	
	@Override
	public Object learn() throws Exception {
		// TODO Auto-generated method stub
		List<int[]> xIndices = new ArrayList<>();
		List<int[]> zIndices = new ArrayList<>();
		AttributeList attList = null;
		
		Profile profile0 = null;
		if (this.sample.next()) {
			profile0 = this.sample.pick();
		}
		this.sample.reset();
		if (profile0 == null) {
			clear();
			return null;
		}
		int n = profile0.getAttCount(); //x1, x2,..., x(n-1), z
		if (n < 2) {
			clear();
			return null;
		}
		attList = profile0.getAttRef();
		xIndices.add(new int[] {-1}); // due to X = (1, x1, x2,..., x(n-1)) and there is no 1 in data.
		zIndices.add(new int[] {-1}); // due to Z = (1, z) and there is no 1 in data.

		//Begin extracting indices from configuration
		List<Integer> indices = new ArrayList<>();
		if (this.getConfig().containsKey(INDICES_FIELD)) {
			String cfgIndices = this.getConfig().getAsString(INDICES_FIELD).trim();
			if (!cfgIndices.isEmpty() && !cfgIndices.contains("-1"))
				indices = TextParserUtil.parseListByClass(cfgIndices, Integer.class, ",");
		}
		if (indices == null || indices.size() < 2) {
			for (int j = 0; j < n - 1; j++)
				xIndices.add(new int[] {j});
			zIndices.add(new int[] {n - 1});
		}
		else {
			for (int j = 0; j < indices.size() - 1; j++)
				xIndices.add(new int[] {indices.get(j)});
			zIndices.add(new int[] {indices.get(indices.size() - 1)}); //The last index is Z index
		}
		if (zIndices.size() < 2 || xIndices.size() < 2) {
			clear();
			return null;
		}
		//End extracting indices from configuration
		
		//Begin checking existence of values.
		boolean zExists = false;
		boolean[] xExists = new boolean[xIndices.size() - 1]; //profile = (x1, x2,..., x(n-1), z)
		Arrays.fill(xExists, false);
		while (this.sample.next()) {
			Profile profile = this.sample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double lastValue = profile.getValueAsReal(zIndices.get(1)[0]);
			if (Util.isUsed(lastValue))
				zExists = zExists || true; 
			
			for (int j = 1; j < xIndices.size(); j++) {
				double value = profile.getValueAsReal(xIndices.get(j)[0]);
				if (Util.isUsed(value))
					xExists[j - 1] = xExists[j - 1] || true;
			}
		}
		this.sample.reset();

		List<int[]> xIndicesTemp = new ArrayList<>();
		xIndicesTemp.add(xIndices.get(0)); //adding -1
		for (int j = 1; j < xIndices.size(); j++) {
			if (xExists[j - 1])
				xIndicesTemp.add(xIndices.get(j)); //only use variables having at least one value.
		}
		if (!zExists || xIndicesTemp.size() < 2) {
			clear();
			return null;
		}
		xIndices = xIndicesTemp;
		//End checking existence of values.

		//Begin extracting data
		n = xIndices.size();
		List<List<Double>> xVectors = Util.newList();
		List<Double> zVector = Util.newList();
		while (this.sample.next()) {
			Profile profile = this.sample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			List<Double> xVector0 = null;
			if (xVectors.size() <= 0) {
				xVector0 = new ArrayList<>();
				xVectors.add(xVector0);
			}
			xVector0 = xVectors.get(0);
			xVector0.add(1.0);
			
			for (int j = 1; j < xIndices.size(); j++) {
				List<Double> xVector = null;
				if (xVectors.size() <= j) {
					xVector = new ArrayList<>();
					xVectors.add(xVector);
				}
				xVector = xVectors.get(j);

				double value = profile.getValueAsReal(xIndices.get(j)[0]);
				xVector.add((double)transformRegressor(value));
			}
			
			double lastValue = profile.getValueAsReal(zIndices.get(1)[0]);
			zVector.add((double)transformResponse(lastValue));
		}
		this.sample.close();
		//End extracting data
		
		this.xVectors.clear();
		this.xVectors = xVectors;
		this.zVector.clear();
		this.zVector = zVector;
		this.xIndices = xIndices;
		this.zIndices = zIndices;
		this.attList = attList;

		this.coeffs = learn0();
		return this.coeffs;
	}


	/**
	 * Internal method to learn coefficients. This method is called by {@link #learn()}.
	 * @return the coefficients to be learned.
	 * @exception Exception if any error occurs.
	 */
	private double[] learn0() throws Exception {
		int n = xIndices.size();
		RealMatrix A = MatrixUtils.createRealMatrix(n, n);
		RealVector b = new ArrayRealVector(n);
		
		for (int i = 0; i < n; i++) {
			double covXiZ = cor(xVectors.get(i), zVector);
			b.setEntry(i, covXiZ);
		}
		
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				double covXiXj = cor(xVectors.get(i), xVectors.get(j));
				A.setEntry(i, j, covXiXj);
			}
		}

		//Due to correlation matrix is symmetric
		for (int i = n-1; i >= 1; i--) {
			for (int j = i-1; j >= 0; j--) {
				double covXiXj = A.getEntry(j, i);
				A.setEntry(i, j, covXiXj);
			}
		}
		
		double[] x = null;
		try {
			DecompositionSolver solver = new QRDecomposition(A).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
			x = solver.solve(b).toArray(); //solve Ax = b with approximation
		}
		catch (SingularMatrixException e) {
			logger.info("Singular matrix problem occurs in #solve(RealMatrix, RealVector)");
			
			//Proposed solution will be improved in next version
			try {
				DecompositionSolver solver = new SingularValueDecomposition(A).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
				RealMatrix pseudoInverse = solver.getInverse();
				x = pseudoInverse.operate(b).toArray();
			}
			catch (SingularMatrixException e2) {
				logger.info("Cannot solve the problem of singluar matrix by Moore–Penrose pseudo-inverse matrix in #solve(RealMatrix, RealVector)");
				x = null;
			}
		}
		
		if (x == null)
			return null;
		for (int i = 0; i < x.length; i++) {
			if (Double.isNaN(x[i]) || !Util.isUsed(x[i]))
				return null;
		}
		
//		double sumZ = 0;
//		int nZ = 0;
//		for (int i = 0; i < zVector.size(); i++) {
//			double z = zVector.get(i) != null ? zVector.get(i) : Constants.UNUSED; 
//			if (Util.isUsed(z)) {
//				sumZ += zVector.get(i);
//				nZ++;
//			}
//		}
//		x[0] = x[0] + sumZ / (double)nZ;
		return x;
	}
	
	
	/**
	 * Calculating the correlation between two vectors. This method will be re-implemented in future.
	 * @param xVector specified vector x.
	 * @param yVector specified vector y.
	 * @return the correlation between two vectors.
	 */
	private double cor(List<Double> xVector, List<Double> yVector) {
		if (xVector.size() == 0 || yVector.size() == 0)
			return 0;
		
		int N = Math.min(xVector.size(), yVector.size());
		List<Integer> U = new ArrayList<>(N);
		double meanX = 0, meanY = 0;
		for (int i = 0; i < N; i++) {
			double x = xVector.get(i) != null ? xVector.get(i) : Constants.UNUSED; 
			double y = yVector.get(i) != null ? yVector.get(i) : Constants.UNUSED; 
			if (!Util.isUsed(x) || !Util.isUsed(y))
				continue;
			
			meanX += x;
			meanY += y;
			U.add(i);
		}
		if (U.size() == 0)
			return 0;
		
		meanX = meanX / (double)U.size();
		meanY = meanY / (double)U.size();
		double cov = 0;
		double varX = 0;
		double varY = 0;
		for (int i = 0; i < U.size(); i++) {
			int k = U.get(i);
			double x = xVector.get(k) - meanX;
			double y = yVector.get(k) - meanY;
			cov += x * y;
			varX += x * x;
			varY += y * y;
		}

		if (varX != 0 && varY != 0) {
			return cov / (Math.sqrt(varX*varY));
		}
		else {
			if (varX == 0 && varY == 0)
				return 1;
			else
				return 0;
		}
	}


	@Override
	public synchronized void unsetup() {
		super.unsetup();
		xVectors.clear();
		zVector.clear();
	}

	
	/**
	 * Clear all internal data.
	 */
	private void clear() {
		unsetup();
		this.coeffs = null;
		this.xIndices.clear();
		this.zIndices.clear();
		this.attList = null;
	}

	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "corr";
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		CorRegression cor = new CorRegression();
		cor.getConfig().putAll((DataConfig)this.getConfig().clone());
		return cor;
	}
	
	
}
