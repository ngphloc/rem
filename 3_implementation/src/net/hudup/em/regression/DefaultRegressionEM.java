package net.hudup.em.regression;

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

import net.hudup.Evaluator;
import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.NextUpdate;
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
	protected final static String REM_INDICES_FIELD = "rem_indices";

	
	/**
	 * Inverse mode field name.
	 */
	protected final static String REM_INVERSE_MODE_FIELD = "rem_inverse_mode";

	
	/**
	 * Inverse mode field name.
	 */
	protected final static String REM_BALANCE_MODE_FIELD = "rem_balance_mode";

	
	/**
	 * Default inverse mode field.
	 */
	protected final static boolean REM_INVERSE_MODE_DEFAULT = false;

	
	/**
	 * Default inverse mode field.
	 */
	protected final static boolean REM_BALANCE_MODE_DEFAULT = true;

	
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
		if (this.getConfig().containsKey(REM_INDICES_FIELD)) {
			String cfgIndices = this.getConfig().getAsString(REM_INDICES_FIELD).trim();
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
		
		//Checking existence of values.
		boolean zExists = false;
		boolean[] xExists = new boolean[xIndices.size() - 1]; //profile = (x1, x2,..., x(n-1), z)
		Arrays.fill(xExists, false);
		while (this.sample.next()) {
			Profile profile = this.sample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double lastValue = profile.getValueAsReal(zIndices.get(1));
			if (Util.isUsed(lastValue))
				zExists = zExists || true; 
			
			for (int j = 1; j < xIndices.size(); j++) {
				double value = profile.getValueAsReal(xIndices.get(j));
				if (Util.isUsed(value))
					xExists[j - 1] = xExists[j - 1] || true;
			}
		}
		this.sample.reset();

		List<Integer> xIndicesTemp = new ArrayList<>();
		xIndicesTemp.add(xIndices.get(0)); //adding -1
		for (int j = 1; j < xIndices.size(); j++) {
			if (xExists[j - 1])
				xIndicesTemp.add(xIndices.get(j)); //only use variables having at least one value.
		}
		if (!zExists || xIndicesTemp.size() < 2) {
			unsetup();
			return null;
		}
		xIndices = xIndicesTemp;
		
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
			if (!getConfig().getAsBoolean(REM_INVERSE_MODE_FIELD))
				stat = estimate(stat, alpha, betas);
			else
				stat = estimateInverse(stat, alpha, betas);
			
			if (stat == null) //Cannot estimate
				return null;
			else {
				zStatistics[i] = stat.getZStatistic();
				xStatistics.add(stat.getXStatistic());
			}
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
		ExchangedParameter currentParameter = (ExchangedParameter) getCurrentParameter();
		
		int N = zStatistics.length;
		int n = xStatistics.get(0).length; //1, x1, x2,..., x(n-1)
		double[] alpha = calcCoeffs(xStatistics, zStatistics);
		if (alpha == null) {
			if (currentParameter != null)
				alpha = Arrays.copyOf(currentParameter.vector, currentParameter.vector.length);
			else { //Used for initialization so that regression model is always determined.
				alpha = new double[n];
				Arrays.fill(alpha, 0.0);
				for (int i = 0; i < N; i++)
					alpha[0] += zStatistics[i];
				alpha[0] = alpha[0] / (double)N; //constant function z = c
			}
		}
		
		List<double[]> betas = new ArrayList<>();
		for (int j = 0; j < n; j++) {
			if (j == 0) {
				double[] beta0 = new double[2];
				beta0[0] = 1;
				beta0[1] = 0;
				betas.add(beta0);
				continue;
			}
			
			List<double[]> Z = new ArrayList<>(N);
			double[] x = new double[N];
			for (int i = 0; i < N; i++) {
				double[] zRow = new double[2];
				Z.add(zRow);
				zRow[0] = 1;
				zRow[1] = zStatistics[i];
				x[i] = xStatistics.get(i)[j];
			}
			double[] beta = calcCoeffs(Z, x);
			if (beta == null) {
				if (currentParameter != null) {
					beta = Arrays.copyOf(currentParameter.matrix.get(j),
						currentParameter.matrix.get(j).length);
				}
				else { //Used for initialization so that regression model is always determined.
					beta = new double[2];
					beta[1] = 0;
					for (int i = 0; i < N; i++)
						beta[0] += xStatistics.get(i)[j];
					beta[0] = beta[0] / (double)N; //constant function x = c
				}
			}
			betas.add(beta);
		}
		
		return new ExchangedParameter(alpha, betas);
	}
	
	
	
	/**
	 * Estimating statistics with specified parameters alpha and beta.
	 * @param stat specified statistics.
	 * @param alpha specified alpha parameter.
	 * @param betas specified alpha parameters.
	 * @return estimated statistics with specified parameters alpha and beta.
	 */
	private Statistics estimate(Statistics stat, double[] alpha, List<double[]> betas) {
		double zValue = stat.getZStatistic();
		double[] xVector = stat.getXStatistic();
		double zStatistic = Constants.UNUSED;
		double[] xStatistic = new double[xVector.length];
		
		if (Util.isUsed(zValue)) {
			zStatistic = zValue;
			
			//Estimating missing xij (xStatistic) by equation 5 and zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
			}
			
			return new Statistics(zStatistic, xStatistic);
		}
		
		//Estimating missing zi (zStatistic) by equation 7, based on current parameter.
		int a = 0, b = 0, c = 0;
		List<Integer> U = new ArrayList<>();
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j])) {
				b += alpha[j] * xVector[j];
			}
			else {
				a += alpha[j] * betas.get(j)[0];
				c += alpha[j] * betas.get(j)[1];
				U.add(j);
			}
		}
		if (c != 1)
			zStatistic = (a + b) / (1 - c);
		else {
			logger.info("Cannot estimate statistic for Z by expectation and so other technique is used");
			zStatistic = zStatisticOtherEstimate(); //Fixing zero denominator
			if (!Util.isUsed(zStatistic)) //Cannot estimate
				return null;
		}
		
		//Estimating missing xij (xStatistic) by equation 5 and estimated zi (zStatistic) above, based on current parameter.
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j]))
				xStatistic[j] = xVector[j];
			else
				xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
		}
		
		return balanceStatistics(alpha, betas, zStatistic, xStatistic, U);
	}

	
	/**
	 * Estimating statistics with specified parameters alpha and beta.
	 * @param stat specified statistics.
	 * @param alpha specified alpha parameter.
	 * @param betas specified alpha parameters.
	 * @return estimated statistics with specified parameters alpha and beta.
	 */
	private Statistics estimateInverse(Statistics stat, double[] alpha, List<double[]> betas) {
		double zValue = stat.getZStatistic();
		double[] xVector = stat.getXStatistic();
		double zStatistic = Constants.UNUSED;
		double[] xStatistic = new double[xVector.length];
		
		if (Util.isUsed(zValue)) {
			zStatistic = zValue;
			//Estimating missing xij (xStatistic) by equation 5 and zi (zStatistic) above, based on current parameter.
			for (int j = 0; j < xVector.length; j++) {
				if (Util.isUsed(xVector[j]))
					xStatistic[j] = xVector[j];
				else
					xStatistic[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
			}
			
			return new Statistics(zStatistic, xStatistic);
		}
		
		List<Integer> U = new ArrayList<>();
		int b = 0;
		for (int j = 0; j < xVector.length; j++) {
			if (Util.isUsed(xVector[j])) {
				b += alpha[j] * xVector[j];
				xStatistic[j] = xVector[j]; //existent xij
			}
			else
				U.add(j);
		}

		if (U.size() > 0) {
			//Estimating missing xij (xStatistic) by equation 8, based on current parameter.
			List<double[]> A = new ArrayList<>(U.size());
			double[] y = new double[U.size()];
			
			for (int i = 0; i < U.size(); i++) {
				double[] aRow = new double[U.size()];
				A.add(aRow);
				for (int j = 0; j < U.size(); j++) {
					if (i == j)
						aRow[j] = betas.get(U.get(i))[1] * alpha[U.get(j)] - 1;
					else
						aRow[j] = betas.get(U.get(i))[1] * alpha[U.get(j)];
				}
				y[i] = -betas.get(U.get(i))[0] - betas.get(U.get(i))[1] * b;
			}
			
			double[] solution = solve(A, y); //solve Ax = y
			if (solution != null) {
				for (int j = 0; j < U.size(); j++)
					xStatistic[U.get(j)] = solution[j]; 
			}
			else {
				logger.info("Cannot estimate statistic for X by expectation and so other technique is used");
				for (int j = 0; j < U.size(); j++) {
					xStatistic[U.get(j)] = xStatisticOtherEstimate(U.get(j));
					if (!Util.isUsed(xStatistic[U.get(j)])) //Cannot estimate
						return null;
				}
			}
		}
		
		//Estimating missing zi (zStatistic) by equation 4, based on current parameter.
		zStatistic = 0;
		for (int j = 0; j < xStatistic.length; j++) {
			zStatistic += alpha[j] * xStatistic[j];
		}
		
		return balanceStatistics(alpha, betas, zStatistic, xStatistic, U);
	}
	
	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		int N = this.zData.size();
		int n = this.xData.get(0).length;
		
		double[] alpha0 = new double[n];
		Arrays.fill(alpha0, 0.0);
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
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		for (int i = 0; i < parameter1.length; i++) {
			if (parameter1[i] == 0) {
				if (parameter2[i] == 0)
					continue;
				else
					return false;
			}
			else {
				double biasRatio = Math.abs(parameter2[i] - parameter1[i]) / Math.abs(parameter1[i]);
				if (biasRatio > threshold)
					return false;
			}
		}
		
		return true;
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
		config.put(REM_INDICES_FIELD, "-1, -1, -1"); //Not used
		config.put(REM_INVERSE_MODE_FIELD, REM_INVERSE_MODE_DEFAULT);
		config.put(REM_BALANCE_MODE_FIELD, REM_BALANCE_MODE_DEFAULT);
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
	protected class ExchangedParameter {
		
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
	 * 
	 */
	protected class Statistics {

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
		
	}
	
	
	/**
	 * Balancing missing values zi (xStatistic) and xij (xValues).
	 * @param alpha alpha coefficients.
	 * @param betas beta coefficients.
	 * @param zStatistic statistic for Z variable.
	 * @param xStatistic statistic for X variables.
	 * @param U list of missing X values.
	 * @return balanced statistics for Z and X variables.
	 */
	private Statistics balanceStatistics(double[] alpha, List<double[]> betas,
			double zStatistic, double[] xStatistic,
			List<Integer> U) {
		if (!getConfig().getAsBoolean(REM_BALANCE_MODE_FIELD))
			return new Statistics(zStatistic, xStatistic);
		
		double zStatisticNext = Constants.UNUSED;
		double[] xStatisticNext = new double[xStatistic.length];
		int t = 0;
		int maxIteration = getConfig().getAsInt(EM_MAX_ITERATION_FIELD);
		maxIteration = (maxIteration <= 0) ? EM_MAX_ITERATION : maxIteration;
		double threshold = getConfig().getAsReal(EM_EPSILON_FIELD);
		while (t < maxIteration) {
			if (!getConfig().getAsBoolean(REM_INVERSE_MODE_FIELD)) {
				zStatisticNext = 0;
				for (int j = 0; j < xStatistic.length; j++)
					zStatisticNext += alpha[j] * xStatistic[j];
				
				for (int j = 0; j < xStatistic.length; j++) {
					if (!U.contains(j))
						xStatisticNext[j] = xStatistic[j];
					else
						xStatisticNext[j] = betas.get(j)[0] + betas.get(j)[1] * zStatisticNext;
				}
			}
			else {
				for (int j = 0; j < xStatistic.length; j++) {
					if (!U.contains(j))
						xStatisticNext[j] = xStatistic[j];
					else
						xStatisticNext[j] = betas.get(j)[0] + betas.get(j)[1] * zStatistic;
				}
				
				zStatisticNext = 0;
				for (int j = 0; j < xStatistic.length; j++)
					zStatisticNext += alpha[j] * xStatisticNext[j];
			}
			
			t++;
			
			//Testing approximation
			boolean approx = true;
			if (zStatistic == 0) {
				if (zStatisticNext == 0)
					approx = true;
				else
					approx = false;
			}
			else
				approx = (Math.abs(zStatisticNext - zStatistic) / Math.abs(zStatistic) <= threshold);
				
			for (int j = 0; j < xStatistic.length; j++) {
				if (xStatistic[j] == 0) {
					if (xStatisticNext[j] == 0)
						approx = approx && true;
					else
						approx = approx && false;
				}
				else
					approx = approx && (Math.abs(xStatisticNext[j] - xStatistic[j]) / Math.abs(xStatistic[j]) <= threshold);
				
				if (!approx) break;
			}
			
			if (approx) break;

			zStatistic = zStatisticNext;
			xStatistic = xStatisticNext;
			zStatisticNext = Constants.UNUSED;
			xStatisticNext = new double[xStatistic.length];
		}
		
		return new Statistics(zStatistic, xStatistic);
	}


	/**
	 * Calculating coefficients based on data matrix and data vector.
	 * This method will be improved in the next version.
	 * @param X specified data matrix.
	 * @param x specified data vector.
	 * @return coefficients base on data matrix and data vector.
	 */
	private double[] calcCoeffs(List<double[]> X, double[] x) {
		int N = x.length;
		int n = X.get(0).length;
		
		List<double[]> A = new ArrayList<>(n);
		for (int i = 0; i < n; i++) {
			double[] aRow = new double[n];
			A.add(aRow);
			for (int j = 0; j < n; j++) {
				double sum = 0;
				for (int k = 0; k < N; k++) {
					sum += X.get(k)[i] * X.get(k)[j];
				}
				aRow[j] = sum;
			}
		}
		
		double[] b = new double[n];
		for (int i = 0; i < n; i++) {
			double sum = 0;
			for (int k = 0; k < N; k++)
				sum += X.get(k)[i] * x[k];
			b[i] = sum;
		}
		
		return solve(A, b);
	}
	

	/**
	 * Estimating the statistic for Z variable (zStatistic) by nearest neighbor filtering.
	 * @return the estimated statistic for Z variable (zStatistic) by nearest neighbor filtering.
	 */
	protected double zStatisticOtherEstimate() {
		List<double[]> source = new ArrayList<>();
		List<Double> target = new ArrayList<>();
		int N = this.zData.size();
		for (int i = 0; i < N; i++) {
			double value = this.zData.get(i)[1]; //due to Z = (1, z)
			if (Util.isUsed(value)) {
				source.add(this.xData.get(i));
				target.add(value);
			}
		}
		
		double[] targetArray = new double[target.size()];
		for (int i = 0; i < target.size(); i++)
			targetArray[i] = target.get(i);
		return nearestNeighborFilter(source, targetArray);
	}
	
	
	/**
	 * Estimating the statistic for X variables (xStatistic) by nearest neighbor filtering.
	 * @param index the index of statistic.
	 * @return the estimated statistic for X variables (xStatistic) by nearest neighbor filtering.
	 */
	protected double xStatisticOtherEstimate(int index) {
		List<double[]> source = new ArrayList<>();
		List<Double> target = new ArrayList<>();
		int N = this.xData.size();
		for (int i = 0; i < N; i++) {
			double value = this.xData.get(i)[index];
			if (Util.isUsed(value)) {
				source.add(new double[] {this.zData.get(i)[1]}); //due to Z = (1, z)
				target.add(value);
			}
		}
		
		double[] targetArray = new double[target.size()];
		for (int i = 0; i < target.size(); i++)
			targetArray[i] = target.get(i);
		return nearestNeighborFilter(source, targetArray);
	}
	
	
	/**
	 * Estimating a variable the statistic nearest neighbor filtering.
	 * The current implementation is average method and so it is enhanced in the next version.
	 * @param source source data.
	 * @param target target data.
	 * @return the estimated variable by nearest neighbor filtering. Return NaN if it is impossible to filter.
	 */
	@NextUpdate
	protected double nearestNeighborFilter(List<double[]> source, double[] target) {
		int N = target.length;
		if (N == 0)
			return 0;
		
		double mean = 0;
		for (int i = 0; i < N; i++)
			mean += target[i];
		
		return mean / (double)N;
	}

	
	/**
	 * Solving the equation Ax = b.
	 * @param A specified matrix.
	 * @param b specified vector.
	 * @return solution x of the equation Ax = b.
	 */
	@NextUpdate
	protected double[] solve(List<double[]> A, double[] b) {
		int N = b.length;
		int n = A.get(0).length;
		if (N == 0 || n == 0)
			return null;
		
		double[] x = null;
		RealMatrix M = MatrixUtils.createRealMatrix(A.toArray(new double[N][n]));
		RealVector m = new ArrayRealVector(b);
		try {
			DecompositionSolver solver = new QRDecomposition(M).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
			x = solver.solve(m).toArray(); //solve Ax = b with approximation
		}
		catch (SingularMatrixException e) {
			logger.info("Singular matrix problem occurs in #solve(RealMatrix, RealVector)");
			
			//Proposed solution will be improved in next version
			try {
				DecompositionSolver solver = new SingularValueDecomposition(M).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
				RealMatrix pseudoInverse = solver.getInverse();
				x = pseudoInverse.operate(m).toArray();
			}
			catch (SingularMatrixException e2) {
				logger.info("Cannot solve the problem of singluar matrix by Moore–Penrose pseudo-inverse matrix in #solve(RealMatrix, RealVector)");
				x = null;
			}
		}
		
		if (x == null)
			return null;
		for (int i = 0; i < x.length; i++) {
			if (Double.isNaN(x[i]))
				return null;
		}
		return x;
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

