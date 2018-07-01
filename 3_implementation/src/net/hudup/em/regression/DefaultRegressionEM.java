package net.hudup.em.regression;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.core.parser.TextParserUtil;
import net.hudup.em.ExponentialEM;


/**
 * This class implements default expectation maximization (EM) algorithm for regression model in case of missing data. 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DefaultRegressionEM extends ExponentialEM implements RegressionEM, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Regression indices field.
	 */
	protected final static String REGRESSION_INDICES = "rem_regression_indices";

	
	/**
	 * Variable contains complete data of X.
	 */
	protected List<double[]> xData = Util.newList(); //1, x1, x2,..., x(n-1)
	
	
	/**
	 * Indices for X data.
	 */
	protected List<Integer> xIndices = new ArrayList<>();
	
	
	/**
	 * Variable contains complete data of Z.
	 */
	protected List<double[]> zData = Util.newList(); //1, z
	
	
	/**
	 * Indices for Z data.
	 */
	protected List<Integer> zIndices = new ArrayList<>();
	
	
	/**
	 * Attribute list for all variables: all X, Y, and z.
	 */
	protected AttributeList attList = null;
	
	
	/**
	 * Default constructor.
	 */
	public DefaultRegressionEM() {
		// TODO Auto-generated constructor stub
	}
	
	
	@Override
	public synchronized Object learn() throws Exception {
		// TODO Auto-generated method stub
		Profile profile0 = null;
		if (this.sample.next()) {
			profile0 = sample.pick();
		}
		this.sample.reset();
		if (profile0 == null) {
			unsetup();
			return null;
		}
		int n = profile0.getAttCount(); //x1, x2,..., x(n-1), z
		if (n < 2) {
			unsetup();
			return null;
		}
		this.attList = profile0.getAttRef();
		this.xIndices.add(-1); // due to X = (1, x1, x2,..., x(n-1)) and there is no 1 in data.
		this.zIndices.add(-1); // due to Z = (1, z) and there is no 1 in data.
		
		List<Integer> indices = new ArrayList<>();
		if (this.getConfig().containsKey(REGRESSION_INDICES))
			indices = TextParserUtil.parseListByClass(this.getConfig().getAsString(REGRESSION_INDICES), Integer.class, ",");
		if (indices == null || indices.size() < 2) {
			for (int j = 0; j < n - 1; j++)
				this.xIndices.add(j);
			this.zIndices.add(n - 1);
		}
		else {
			for (int j = 0; j < indices.size() - 1; j++)
				this.xIndices.add(indices.get(j));
			this.zIndices.add(indices.get(indices.size() - 1));
		}
		if (this.zIndices.size() < 2 || this.xIndices.size() < 2) {
			unsetup();
			return null;
		}
		
		n = xIndices.size();
		while (this.sample.next()) {
			Profile profile = sample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double[] xVector = new double[n]; //1, x1, x2,..., x(n-1)
			double[] zVector = new double[2]; //1, z
			xVector[0] = 1;
			zVector[0] = 1;
			
			boolean zExist = false;
			double lastValue = profile.getValueAsReal(zIndices.get(1));
			if (!Util.isUsed(lastValue))
				zVector[1] = Constants.UNUSED;
			else {
				zVector[1] = lastValue;
				zExist = true;
			}
			
			boolean xExist = false;
			for (int j = 1; j < xIndices.size(); j++) {
				double value = profile.getValueAsReal(xIndices.get(j));
				if (!Util.isUsed(value))
					xVector[j] = Constants.UNUSED;
				else {
					xVector[j] = value;
					xExist = true;
				}
			}
			
			if(zExist || xExist) {
				zData.add(zVector);
				xData.add(xVector);
			}
		}
		this.sample.close();
		
		if (xData.size() == 0 || zData.size() == 0) {
			unsetup();
			return null;
		}
		
		return super.learn();
	}


	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		xData.clear();
		xIndices.clear();
		
		zData.clear();
		zIndices.clear();
		
		this.attList = null;
	}


	@Override
	protected Object expectation(Object currentParameter) throws Exception {
		// TODO Auto-generated method stub
		double[] alpha = ((InternalParameter)currentParameter).getArray();
		List<double[]> betas = ((InternalParameter)currentParameter).getArrayList();

		int N = this.zData.size();
		double[] zStatistic = new double[N];
		List<double[]> xStatistics = new ArrayList<>();
		int n = this.xData.get(0).length; //1, x1, x2,..., x(n-1)
		for (int i = 0; i < N; i++) {
			double[] xStatistic = new double[n];
			xStatistics.add(xStatistic);
		}
		
		for (int i = 0; i < N; i++) {
			double[] x = this.xData.get(i);
			double value = this.zData.get(i)[1];
			List<Integer> U = new ArrayList<>();
			if (Util.isUsed(value)) {
				zStatistic[i] = value;
			}
			else {
				int a = 0, b = 0, c = 0;
				for (int j = 0; j < x.length; j++) {
					if (Util.isUsed(x[j])) {
						b += alpha[j] * x[j];
					}
					else {
						a += alpha[j] * betas.get(j)[0];
						c += alpha[j] * betas.get(j)[1];
						U.add(j);
					}
				}
				if (c != 1)
					zStatistic[i] = (a + b) / (1 - c);
				else // Fixing zero denominator
					zStatistic[i] = 0;
			}
			
			double[] xStatistic = xStatistics.get(i);
			for (int j = 0; j < x.length; j++) {
				if (Util.isUsed(x[j])) {
					xStatistic[j] = x[j];
				}
				else {
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic[i];
				}
			}
		}
		
		return new InternalParameter(zStatistic, xStatistics);
	}

	
	@Override
	protected Object maximization(Object currentStatistic) throws Exception {
		// TODO Auto-generated method stub
		double[] zStatistic = ((InternalParameter)currentStatistic).getArray();
		List<double[]> xStatistics = ((InternalParameter)currentStatistic).getArrayList();
		if (zStatistic.length == 0 || xStatistics.size() <= 1)
			return null;
		
		int N = zStatistic.length;
		int n = xStatistics.size(); //1, x1, x2,..., x(n-1)
		RealMatrix X = MatrixUtils.createRealMatrix(xStatistics.toArray(new double[N][n]));
		RealVector z = new ArrayRealVector(N);
		for (int i = 0; i < N; i++) {
			z.setEntry(i, zStatistic[i]);
		}
		RealMatrix Xt = X.transpose();
		double[] alpha = MatrixUtils.inverse(Xt.multiply(X)).multiply(Xt).operate(z).
					toArray();
		
		List<double[]> betas = ((InternalParameter)currentParameter).getArrayList();
		for (int j = 0; j < n; j++) {
			RealMatrix Z = MatrixUtils.createRealMatrix(N, 2);
			RealVector x = new ArrayRealVector(N);
			for (int i = 0; i < N; i++) {
				Z.setEntry(i, 0, 1);
				Z.setEntry(i, 1, zStatistic[i]);
				x.setEntry(i, xStatistics.get(j)[i]);
			}
			RealMatrix Zt = Z.transpose();
			double[] beta = MatrixUtils.inverse(Zt.multiply(Z)).multiply(Zt).operate(x).
						toArray();
			betas.add(beta);
		}
		
		return new InternalParameter(alpha, betas);
	}
	
	
	/**
	 * This class represents the internal parameter of this EM algorithm.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	protected class InternalParameter {
		
		/**
		 * Alpha parameter.
		 */
		double[] array = new double[0];
		
		/**
		 * List of beta parameters
		 */
		List<double[]> arrayList = new ArrayList<>();
		
		/**
		 * Constructor with specified array and array list.
		 * @param array specified array.
		 * @param arrayList specified array list.
		 */
		public InternalParameter(double[] array, List<double[]> arrayList) {
			this.array = array;
			this.arrayList = arrayList;
		}
		
		/**
		 * Getting array parameter.
		 * @return array parameter.
		 */
		public double[] getArray() {
			return array;
		}
		
		/**
		 * Getting array list.
		 * @return array list.
		 */
		public List<double[]> getArrayList() {
			return arrayList;
		}
	}


	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		int N = this.zData.size();
		int n = this.xData.get(0).length;
		
		double[] alpha0 = new double[N];
		for (int i = 0; i < N; i++)
			alpha0[i] = 0;
		List<double[]> betas0 = new ArrayList<>();
		for (int j = 0; j < n; j++) {
			double[] beta0 = new double[2];
			beta0[0] = 0;
			beta0[1] = 0;
			betas0.add(beta0);
		}
		InternalParameter parameter0 = new InternalParameter(alpha0, betas0);
		
		List<Double> zValues = new ArrayList<>();
		List<double[]> xStatistics = new ArrayList<>();
		for (int i = 0; i < N; i++) {
			double zValue = this.zData.get(i)[1];
			if (!Util.isUsed(zValue))
				continue;
			
			double[] xVector = this.xData.get(i);
			boolean missing = false;
			for (int j = 0; j < xVector.length; j++) {
				if (!Util.isUsed(xVector[j])) {
					missing = true;
					break;
				}
			}
			
			if (!missing) {
				zValues.add(zValue);
				xStatistics.add(xVector);
			}
		}
		
		if (zValues.size() == 0)
			return parameter0;
		
		N = zValues.size();
		double[] zStatistic = new double[N];
		for (int i = 0; i < N; i++)
			zStatistic[i] = zValues.get(i);
		InternalParameter currentStatistic = new InternalParameter(zStatistic, xStatistics);
		try {
			return maximization(currentStatistic);
		}
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return parameter0;
	}

	
	@Override
	protected boolean terminatedCondition(Object currentParameter, Object estimatedParameter, Object... info) {
		// TODO Auto-generated method stub
		double[] parameter1 = ((InternalParameter)currentParameter).getArray();
		double[] parameter2 = ((InternalParameter)estimatedParameter).getArray();
		double maxBias = 0;
		for (int i = 0; i < parameter1.length; i++) {
			double bias = Math.abs(parameter1[i] - parameter2[i]);
			if (maxBias < bias)
				maxBias = bias;
		}
		
		return maxBias <= getConfig().getAsReal(EM_EPSILON_FIELD);
	}

	
	@Override
	public Object execute(Object input) {
		// TODO Auto-generated method stub
		double[] alpha = ((InternalParameter)this.estimatedParameter).getArray();
		if (alpha == null || alpha.length == 0)
			return Constants.UNUSED;
		
		if (input == null || !(input instanceof Profile))
			return null; //only support profile input currently
		Profile profile = (Profile)input;
		
		double sum = alpha[0];
		for (int i = 0; i < alpha.length - 1; i++) {
			double value = profile.getValueAsReal(i);
			sum += alpha[i + 1] * value; 
		}
		
		return sum;
	}
	
	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "rem.default";
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		DefaultRegressionEM em = new DefaultRegressionEM();
		em.getConfig().putAll((DataConfig)this.getConfig().clone());
		return em;
	}


	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		return config;
	}

	
	@Override
	public String getDescription() {
		// TODO Auto-generated method stub
		double[] alpha = ((InternalParameter)this.estimatedParameter).getArray();
		if (alpha == null || alpha.length == 0)
			return "";

		StringBuffer buffer = new StringBuffer();
		buffer.append(this.attList.get(this.zIndices.get(this.zIndices.size() - 1)).getName()
				+ " = " + alpha[0]);
		for (int i = 0; i < alpha.length - 1; i++) {
			double coeff = alpha[i + 1];
			String variableName = this.attList.get(this.xIndices.get(i + 1)).getName();
			if (coeff < 0)
				buffer.append(" - " + Math.abs(coeff) + "*" + variableName);
			else
				buffer.append(" + " + coeff + "*" + variableName);
		}
		
		return buffer.toString();
	}


	@Override
	public String parameterToShownText(Object parameter) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof InternalParameter))
			return "";
		double[] array = ((InternalParameter)parameter).getArray();
		return TextParserUtil.toText(array, ",");
	}

	
}
