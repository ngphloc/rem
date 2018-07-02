package net.hudup.em.regression;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import net.hudup.Evaluator;
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
 * This class implements default expectation maximization algorithm for regression model in case of missing data, called REM algorithm. 
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
	protected final static String REM_INDICES = "rem_indices";

	
	/**
	 * Variable contains complete data of X.
	 */
	protected List<double[]> xData = Util.newList(); //1, x1, x2,..., x(n-1)
	
	
	/**
	 * Variable contains complete data of Z.
	 */
	protected List<double[]> zData = Util.newList(); //1, z
	
	
	/**
	 * Indices for X data.
	 */
	protected List<Integer> xIndices = new ArrayList<>();
	
	
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
		List<double[]> xData = Util.newList(); //1, x1, x2,..., x(n-1)
		List<double[]> zData = Util.newList(); //1, z
		List<Integer> xIndices = new ArrayList<>();
		List<Integer> zIndices = new ArrayList<>();
		AttributeList attList = null;
		
		Profile profile0 = null;
		if (this.sample.next()) {
			profile0 = this.sample.pick();
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
		attList = profile0.getAttRef();
		xIndices.add(-1); // due to X = (1, x1, x2,..., x(n-1)) and there is no 1 in data.
		zIndices.add(-1); // due to Z = (1, z) and there is no 1 in data.
		
		List<Integer> indices = new ArrayList<>();
		if (this.getConfig().containsKey(REM_INDICES)) {
			String cfgIndices = this.getConfig().getAsString(REM_INDICES).trim();
			if (!cfgIndices.isEmpty() && !cfgIndices.contains("-1"))
				indices = TextParserUtil.parseListByClass(cfgIndices, Integer.class, ",");
		}
		if (indices == null || indices.size() < 2) {
			for (int j = 0; j < n - 1; j++)
				xIndices.add(j);
			zIndices.add(n - 1);
		}
		else {
			for (int j = 0; j < indices.size() - 1; j++)
				xIndices.add(indices.get(j));
			zIndices.add(indices.get(indices.size() - 1)); //The last index is Z index
		}
		if (zIndices.size() < 2 || xIndices.size() < 2) {
			unsetup();
			return null;
		}
		
		n = xIndices.size();
		while (this.sample.next()) {
			Profile profile = this.sample.pick(); //profile = (x1, x2,..., x(n-1), z)
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
		
		this.xData.clear();
		this.xData = xData; //1, x1, x2,..., x(n-1)
		this.zData.clear();
		this.zData = zData; //1, z
		this.xIndices = xIndices;
		this.zIndices = zIndices;
		this.attList = attList;
		
		return super.learn();
	}


	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		this.xData.clear();
		this.zData.clear();
	}


	@Override
	protected Object expectation(Object currentParameter) throws Exception {
		// TODO Auto-generated method stub
		double[] alpha = ((ExchangedParameter)currentParameter).getVector();
		List<double[]> betas = ((ExchangedParameter)currentParameter).getMatrix();

		int N = this.zData.size();
		double[] zStatistics = new double[N];
		List<double[]> xStatistics = new ArrayList<>();
		for (int i = 0; i < N; i++) {
			Statistics stat = new Statistics(this.zData.get(i)[1], this.xData.get(i));
			stat = stat.estimate(alpha, betas);
			
			zStatistics[i] = stat.getZStatistic();
			xStatistics.add(stat.getXStatistic());
		}
		
		return new ExchangedParameter(zStatistics, xStatistics);
	}

	
	@Override
	protected Object maximization(Object currentStatistic) throws Exception {
		// TODO Auto-generated method stub
		double[] zStatistics = ((ExchangedParameter)currentStatistic).getVector();
		List<double[]> xStatistics = ((ExchangedParameter)currentStatistic).getMatrix();
		if (zStatistics.length == 0 || xStatistics.size() != zStatistics.length)
			return null;
		
		int N = zStatistics.length;
		int n = xStatistics.get(0).length; //1, x1, x2,..., x(n-1)
		RealMatrix X = MatrixUtils.createRealMatrix(xStatistics.toArray(new double[N][n]));
		RealVector z = new ArrayRealVector(zStatistics);
		RealMatrix Xt = X.transpose();
		double[] alpha = MatrixUtils.inverse(Xt.multiply(X)).multiply(Xt).operate(z).
					toArray();
		
		List<double[]> betas = new ArrayList<>();
		for (int j = 0; j < n; j++) {
			if (j == 0) {
				double[] beta0 = new double[2];
				beta0[0] = 1;
				beta0[1] = 0;
				betas.add(beta0);
				continue;
			}
			
			RealMatrix Z = MatrixUtils.createRealMatrix(N, 2);
			RealVector x = new ArrayRealVector(N);
			for (int i = 0; i < N; i++) {
				Z.setEntry(i, 0, 1);
				Z.setEntry(i, 1, zStatistics[i]);
				x.setEntry(i, xStatistics.get(i)[j]);
			}
			RealMatrix Zt = Z.transpose();
			double[] beta = MatrixUtils.inverse(Zt.multiply(Z)).multiply(Zt).operate(x).
						toArray();
			betas.add(beta);
		}
		
		return new ExchangedParameter(alpha, betas);
	}
	
	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		int N = this.zData.size();
		int n = this.xData.get(0).length;
		
		double[] alpha0 = new double[n];
		for (int j = 0; j < n; j++)
			alpha0[j] = 0;
		List<double[]> betas0 = new ArrayList<>();
		for (int j = 0; j < n; j++) {
			double[] beta0 = new double[2];
			if (j == 0) {
				beta0[0] = 1;
				beta0[1] = 0;
			}
			else {
				beta0[0] = 0;
				beta0[1] = 0;
			}
			betas0.add(beta0);
		}
		ExchangedParameter parameter0 = new ExchangedParameter(alpha0, betas0);
		
		List<Double> zVector = new ArrayList<>();
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
				zVector.add(zValue);
				xStatistics.add(xVector);
			}
		}
		
		if (zVector.size() == 0)
			return parameter0;
		
		N = zVector.size();
		double[] zStatistics = new double[N];
		for (int i = 0; i < N; i++)
			zStatistics[i] = zVector.get(i);
		ExchangedParameter currentStatistic = new ExchangedParameter(zStatistics, xStatistics);
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
		double[] parameter1 = ((ExchangedParameter)currentParameter).getVector();
		double[] parameter2 = ((ExchangedParameter)estimatedParameter).getVector();
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
		double[] alpha = ((ExchangedParameter)this.estimatedParameter).getVector();
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
			return "rem";
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
		config.put(REM_INDICES, "-1, -1, -1"); //Not used
		return config;
	}

	
	@Override
	public String getDescription() {
		// TODO Auto-generated method stub
		if (this.getParameter() == null)
			return "";
		double[] alpha = ((ExchangedParameter)this.getParameter()).getVector();
		if (alpha.length == 0)
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
	public String parameterToShownText(Object parameter, Object...info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof ExchangedParameter))
			return "";
		double[] array = ((ExchangedParameter)parameter).getVector();
		return TextParserUtil.toText(array, ",");
	}

	
	/**
	 * This class represents the exchanged parameter for this REM algorithm.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static class ExchangedParameter {
		
		/**
		 * Vector parameter.
		 */
		protected double[] vector = new double[0];
		
		/**
		 * Matrix parameter
		 */
		protected List<double[]> matrix = new ArrayList<>();
		
		/**
		 * Constructor with specified vector and matrix.
		 * @param vector specified vector.
		 * @param matrix specified matrix.
		 */
		public ExchangedParameter(double[] vector, List<double[]> matrix) {
			this.vector = vector;
			this.matrix = matrix;
		}
		
		/**
		 * Getting vector parameter.
		 * @return vector parameter.
		 */
		public double[] getVector() {
			return vector;
		}
		
		/**
		 * Getting matrix.
		 * @return matrix.
		 */
		public List<double[]> getMatrix() {
			return matrix;
		}
		
	}
	
	
	/**
	 * This class represents a compound statistic.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	protected static class Statistics {
		
		/**
		 * Statistic for Z variable.
		 */
		protected double zStatistic = Constants.UNUSED;
		
		/**
		 * Statistic for X variables.
		 */
		protected double[] xStatistic = new double[0];
		
		/**
		 * Constructor with specified statistic for Z variable and statistic for X variables.
		 * @param zStatistic statistic for Z variable.
		 * @param xStatistic statistic for X variables.
		 */
		public Statistics(double zStatistic, double[] xStatistic) {
			this.zStatistic = zStatistic;
			this.xStatistic = xStatistic;
		}
		
		/**
		 * Getting statistic for Z variable.
		 * @return statistic for Z variable.
		 */
		public double getZStatistic() {
			return zStatistic;
		}
		
		/**
		 * Getting statistic for X variables.
		 * @return statistic for X variables.
		 */
		public double[] getXStatistic() {
			return xStatistic;
		}
		
		/**
		 * Estimating statistics with specified parameters alpha and beta.
		 * @param alpha specified alpha parameter.
		 * @param betas specified alpha parameters.
		 * @return estimated statistics with specified parameters alpha and beta.
		 */
		public Statistics estimate(double[] alpha, List<double[]> betas) {
			double zValue = this.getZStatistic();
			double[] xVector = this.getXStatistic();
			double zStatistic = Constants.UNUSED;
			double[] xStatistic = new double[xVector.length];
			
			if (Util.isUsed(zValue)) {
				zStatistic = zValue;
			}
			else {
				//Estimating missing zi (zStatistic) by equation 7, based on current parameter.
				int a = 0, b = 0, c = 0;
				for (int j = 0; j < xVector.length; j++) {
					if (Util.isUsed(xVector[j])) {
						b += alpha[j] * xVector[j];
					}
					else {
						a += alpha[j] * betas.get(j)[0];
						c += alpha[j] * betas.get(j)[1];
					}
				}
				if (c != 1)
					zStatistic = (a + b) / (1 - c);
				else
					zStatistic = 0; //Fixing zero denominator
			}
			
			//Estimating missing xij (xStatistic) by equation 5 and estimated zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
			}
			
			//Re-estimating missing zi (zStatistic) by equation 4 and estimated xij (xStatistic) above, based on current parameter.
			if (!Util.isUsed(zValue)) {
				zStatistic = 0;
				for (int j = 0; j < xStatistic.length; j++)
					zStatistic += alpha[j] * xStatistic[j];
			}
			
			return new Statistics(zStatistic, xStatistic);
		}

	}
	
	
	/**
	 * The main method to start evaluator.
	 * @param args The argument parameter of main method. It contains command line arguments.
	 * @throws Exception if there is any error.
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		new Evaluator().run(args);
	}
	
	
}
