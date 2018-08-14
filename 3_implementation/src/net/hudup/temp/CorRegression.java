package net.hudup.temp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.NextUpdate;
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
	protected List<List<Double>> xVectors = new ArrayList<>();
	
	
	/**
	 * Z vector.
	 */
	protected List<Double> zVector = new ArrayList<>();

	
	/**
	 * Default constructor.
	 */
	public CorRegression() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	
	@Override
	public synchronized Object learn() throws Exception {
		if (!prepareInternalData())
			return null;
		
		int n = xIndices.size();
		List<double[]> A = new ArrayList<>(n);
		double[] b = new double[n];
		
		for (int i = 0; i < n; i++) {
			double covXiZ = cor(xVectors.get(i), zVector);
			b[i] = covXiZ;
			
			double[] vector = new double[n];
			A.add(vector);
		}
		
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				double covXiXj = cor(xVectors.get(i), xVectors.get(j));
				A.get(i)[j] = covXiXj;
			}
		}

		//Due to correlation matrix is symmetric
		for (int i = n-1; i >= 1; i--) {
			for (int j = i-1; j >= 0; j--) {
				A.get(i)[j] = A.get(j)[i];
			}
		}
		
		this.coeffs = AbstractRegression.solve(A, b);
		if (this.coeffs == null)
			return null;
		
		//Adjusting intercept, improved later
		double sumZ = 0;
		int nZ = 0;
		for (int i = 0; i < zVector.size(); i++) {
			double z = zVector.get(i) != null ? zVector.get(i) : Constants.UNUSED; 
			if (Util.isUsed(z)) {
				sumZ += zVector.get(i);
				nZ++;
			}
		}
		this.coeffs[0] = this.coeffs[0] + sumZ / (double)nZ;
		//Adjusting intercept, improved later
		
		return this.coeffs;
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

	
	/**
	 * Preparing data.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData() throws Exception {
		// TODO Auto-generated method stub
		clearInternalData();
		
		Profile profile0 = null;
		if (this.sample.next()) {
			profile0 = this.sample.pick();
		}
		this.sample.reset();
		if (profile0 == null)
			return false;
		if (profile0.getAttCount() < 2) //x1, x2,..., x(n-1), z
			return false;
		this.attList = profile0.getAttRef();
		AbstractRegression.standardizeAttributeNames(this.attList);
		
		String cfgIndices = null;
		if (this.getConfig().containsKey(R_INDICES_FIELD))
			cfgIndices = this.getConfig().getAsString(R_INDICES_FIELD).trim();
		if (!AbstractRegression.parseIndices(cfgIndices, profile0.getAttCount(), this.xIndices, this.zIndices)) { //parsing indices
			clearInternalData();
			return false;
		}

		//Begin checking existence of values.
		boolean zExists = false;
		boolean[] xExists = new boolean[xIndices.size() - 1]; //profile = (x1, x2,..., x(n-1), z)
		Arrays.fill(xExists, false);
		while (this.sample.next()) {
			Profile profile = this.sample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double lastValue = extractResponse(profile);
			if (Util.isUsed(lastValue))
				zExists = zExists || true; 
			
			for (int j = 1; j < xIndices.size(); j++) {
				double value = extractRegressor(profile, j);
				if (Util.isUsed(value))
					xExists[j - 1] = xExists[j - 1] || true;
			}
		}
		this.sample.reset();

		List<Object[]> xIndicesTemp = new ArrayList<>();
		xIndicesTemp.add(xIndices.get(0)); //adding -1
		for (int j = 1; j < xIndices.size(); j++) {
			if (xExists[j - 1])
				xIndicesTemp.add(xIndices.get(j)); //only use variables having at least one value.
		}
		if (!zExists || xIndicesTemp.size() < 2) {
			clearInternalData();
			return false;
		}
		xIndices = xIndicesTemp;
		//End checking existence of values.

		//Begin extracting data
		while (this.sample.next()) {
			Profile profile = this.sample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			List<Double> xVector0 = null;
			if (this.xVectors.size() <= 0) {
				xVector0 = new ArrayList<>();
				this.xVectors.add(xVector0);
			}
			xVector0 = this.xVectors.get(0);
			xVector0.add(1.0);
			
			for (int j = 1; j < this.xIndices.size(); j++) {
				List<Double> xVector = null;
				if (this.xVectors.size() <= j) {
					xVector = new ArrayList<>();
					this.xVectors.add(xVector);
				}
				xVector = this.xVectors.get(j);

				double value = extractRegressor(profile, j);
				xVector.add((double)transformRegressor(value, false));
			}
			
			double lastValue = extractResponse(profile);
			this.zVector.add((double)transformResponse(lastValue, false));
		}
		this.sample.close();
		//End extracting data
		
		return true;
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
	protected void clearInternalData() {
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


	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}
	
	
}
