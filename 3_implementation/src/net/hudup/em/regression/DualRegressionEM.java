package net.hudup.em.regression;

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
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Profile;
import net.hudup.core.parser.TextParserUtil;
import net.hudup.em.ExponentialEM;


/**
 * This class implements expectation maximization (EM) algorithm for two dual regression models when the response variable of the second model can be missing. 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DualRegressionEM extends ExponentialEM implements RegressionEM, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of first execution model field.
	 */
	protected final static String EXECUTION_FIRST_MODE_FIELD = "drem_execution_first_model";
	
	
	/**
	 * Default first execution model.
	 */
	protected final static boolean EXECUTION_FIRST_MODE_DEFAULT = true;
	
	
	/**
	 * Name of dual field.
	 */
	protected final static String DREM_DUAL_FIELD = "drem_dual";
	
	
	/**
	 * The default dual model. If true, the algorithm is learned in dual mode.
	 */
	protected final static boolean DREM_DUAL_DEFAULT = true;

	
	/**
	 * Name of the number of xi variables, stored in configuration.
	 */
	protected final static String DREM_X_NUMBER_FIELD = "drem_x_number";
	
	
	/**
	 * The default number of xi variables, stored in configuration.
	 */
	protected final static int DREM_X_NUMBER_DEFAULT = 0;
	
	
	/**
	 * Name of the number of yi variables, stored in configuration.
	 */
	protected final static String DREM_Y_NUMBER_FIELD = "drem_y_number";
	
	
	/**
	 * The default number of yi variables, stored in configuration.
	 */
	protected final static int DREM_Y_NUMBER_DEFAULT = 0;
	
	
	/**
	 * The number of independent X. It is initialized in the {@link #setup(Dataset, Object...)} method. 
	 * This number can be zero.
	 */
	protected int n = 0;

	
	/**
	 * The number of independent Y. It is initialized in the {@link #setup(Dataset, Object...)} method.
	 * This number can be zero. 
	 */
	protected int k = 0;
	
	
	/**
	 * Variable contains complete data of X.
	 */
	protected List<double[]> xData = Util.newList();
	
	
	/**
	 * Variable contains complete data of X.
	 */
	protected List<double[]> yData = Util.newList();
	
	
	/**
	 * Indices of mapping of X and Y data.
	 */
	protected List<int[]> indices = Util.newList();
	
	
	/**
	 * Attribute list for all variables: all X, Y, and z.
	 */
	protected AttributeList attList = null;
	
	
	/**
	 * Default constructor.
	 */
	public DualRegressionEM() {
		// TODO Auto-generated constructor stub
	}

	
	@Override
	public synchronized Object learn() throws Exception {
		// TODO Auto-generated method stub
		try {
			//n is number of independent variables X
			this.n = getConfig().getAsInt(DREM_X_NUMBER_FIELD);
			//k is number of independent variables Y
			this.k = getConfig().getAsInt(DREM_Y_NUMBER_FIELD);
			
			int k0 = 0, n0 = 0;
			if (this.sample.next()) {
				Profile profile = sample.pick();
				this.attList = profile.getAttRef();
				k0 = 0;
				n0 = this.attList.size() - k0 - 1;
			}
			this.sample.reset();
			
			if (this.n == 0 && this.k == 0) {
				this.n = n0;
				this.k = k0;
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			unsetup();
			return null;
		}
		if (this.n < 0 || this.k < 0 || this.n + this.k >= this.attList.size()) {
			unsetup();
			return null;
		}
		
		//Extract x data
		List<Profile> profileList = Util.newList();
		while (this.sample.next()) {
			//profile = (x0, x1,..., x(n-1), y0, y1,..., y(k-1), z)
			Profile profile = sample.pick();
			if (profile == null)
				continue;
			profileList.add(profile);
			
			double[] vector = new double[this.n + 1]; //x0, x1,..., x(n-1), z
			if (profile.isMissing(profile.getAttCount() - 1))
				vector[this.n] = Constants.UNUSED;
			else {
				double lastValue = profile.getValueAsReal(profile.getAttCount() - 1);
				if (!Util.isUsed(lastValue))
					vector[this.n] = Constants.UNUSED;
				else
					vector[this.n] = lastValue;
			}
			
			int[] index = new int[] {-1, -1};
			this.indices.add(index);
			
			boolean missing = false;
			for (int j = 0; j < this.n; j++) {
				if (profile.isMissing(j)) {
					missing = true;
					break;
				}
				double value = profile.getValueAsReal(j);
				if (!Util.isUsed(value)) {
					missing = true;
					break;
				}
				vector[j] = value;
			}
			
			// x values must be complete
			if (!missing) { 
				this.xData.add(vector);
				index[0] = this.xData.size() - 1;
			}
		}
		this.sample.close();
		
		//Extract Y data
		for (int i = 0; i < profileList.size(); i++) {
			//profile = (x0, x1,..., x(n-1), y0, y1,..., y(k-1), z)
			Profile profile = profileList.get(i);
			double[] vector = new double[this.k + 1]; //y0, y1,..., y(k-1), z
			if (profile.isMissing(profile.getAttCount() - 1))
				vector[this.k] = Constants.UNUSED;
			else {
				double lastValue = profile.getValueAsReal(profile.getAttCount() - 1);
				if (!Util.isUsed(lastValue))
					vector[this.k] = Constants.UNUSED;
				else
					vector[this.k] = lastValue;
			}
			
			boolean missing = false;
			for (int j = 0; j < this.k; j++) {
				if (profile.isMissing(this.n + j)) {
					missing = true;
					break;
				}
				double value = profile.getValueAsReal(this.n + j);
				if (!Util.isUsed(value)) {
					missing = true;
					break;
				}
				vector[j] = value;
			}
			
			int[] index = this.indices.get(i);
			
			// y values must be complete
			if (!missing) {
				this.yData.add(vector);
				index[1] = this.yData.size() - 1;
			}
		}
		profileList.clear();
		
		return super.learn();
	}


	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		n = 0;
		k = 0;
		xData.clear();
		yData.clear();
		indices.clear();
		attList = null;
	}


	@Override
	protected synchronized Object expectation(Object currentParameter) throws Exception {
		// TODO Auto-generated method stub
		RealVector alpha = extractAlpha((double[])currentParameter);
		RealVector beta = extractBeta((double[])currentParameter);

		//Calculating X statistic
		List<double[]> currentStatistic1 = Util.newList();
		for (int i = 0; i < this.xData.size(); i++) {
			double[] profile = this.xData.get(i);
			double[] newProfile = new double[profile.length];
			for (int j = 0; j < profile.length; j++) newProfile[j] = profile[j];
			currentStatistic1.add(newProfile);
			
			if (Util.isUsed(profile[profile.length - 1])) continue;
			
			RealVector x = new ArrayRealVector(profile.length);
			x.setEntry(0, 1);
			for (int j = 0; j < profile.length - 1; j++)
				x.setEntry(1 + j, profile[j]);
			
			newProfile[profile.length - 1] = alpha.dotProduct(x);
		}
		
		//Calculating Y statistic
		List<double[]> currentStatistic2 = Util.newList();
		for (int i = 0; i < this.yData.size(); i++) {
			double[] profile = this.yData.get(i);
			double[] newProfile = new double[profile.length];
			for (int j = 0; j < profile.length; j++) newProfile[j] = profile[j];
			currentStatistic2.add(newProfile);
			
			if (Util.isUsed(profile[profile.length - 1])) continue;
			
			RealVector y = new ArrayRealVector(profile.length);
			y.setEntry(0, 1);
			for (int j = 0; j < profile.length - 1; j++)
				y.setEntry(1 + j, profile[j]);
			
			newProfile[profile.length - 1] = beta.dotProduct(y);
		}

		boolean dualMode = getConfig().getAsBoolean(DREM_DUAL_FIELD);
		if (!dualMode)
			return new Object[] {currentStatistic1, currentStatistic2};
		
		//Calculate Z in dual mode.
		for (int i = 0; i < this.indices.size(); i++) {
			int[] index = this.indices.get(i);
			if (index[0] ==  -1 || index[1] == -1)
				continue;
			
			if(Util.isUsed(this.xData.get(index[0])[this.n]) || Util.isUsed(this.yData.get(index[1])[this.k]))
				continue;
			
			double z1 = currentStatistic1.get(index[0])[this.n];
			double z2 = currentStatistic2.get(index[1])[this.k];
			double z = (z1 + z2) / 2;
			currentStatistic1.get(index[0])[this.n] = z;
			currentStatistic2.get(index[1])[this.k] = z;
		}
		
		return new Object[] {currentStatistic1, currentStatistic2};
	}

	
	@Override
	protected Object maximization(Object currentStatistic) throws Exception {
		// TODO Auto-generated method stub
		@SuppressWarnings("unchecked")
		List<double[]> currentStatistic1 = (List<double[]>) (((Object[])currentStatistic)[0]);
		@SuppressWarnings("unchecked")
		List<double[]> currentStatistic2 = (List<double[]>) (((Object[])currentStatistic)[1]);
		
		//Estimate first model
		RealVector alpha = new ArrayRealVector(this.n + 1);
		alpha.set(0);
		if (currentStatistic1.size() > 0) {
			RealMatrix X = MatrixUtils.createRealMatrix(currentStatistic1.size(), this.n + 1);
			RealVector z1 = new ArrayRealVector(currentStatistic1.size());
			for (int i = 0; i < currentStatistic1.size(); i++) {
				double[] profile = currentStatistic1.get(i);
				
				X.setEntry(i, 0, 1.0);
				for (int j = 0; j < this.n; j++) {
					X.setEntry(i, 1 + j, profile[j]);
				}
				
				z1.setEntry(i, profile[this.n]);
			}
			RealMatrix Xt = X.transpose();
			alpha = MatrixUtils.inverse(Xt.multiply(X)).multiply(Xt).operate(z1);
		}
		
		//Estimate second model
		RealVector beta = new ArrayRealVector(this.k + 1);
		beta.set(0);
		if (currentStatistic2.size() > 0) {
			RealMatrix Y = MatrixUtils.createRealMatrix(currentStatistic2.size(), this.k + 1);
			RealVector z2 = new ArrayRealVector(currentStatistic2.size());
			for (int i = 0; i < currentStatistic2.size(); i++) {
				double[] profile = currentStatistic2.get(i);
				
				Y.setEntry(i, 0, 1.0);
				for (int j = 0; j < this.k; j++) {
					Y.setEntry(i, 1 + j, profile[j]);
				}
	
				z2.setEntry(i, profile[this.k]);
			}
			RealMatrix Yt = Y.transpose();
			beta = MatrixUtils.inverse(Yt.multiply(Y)).multiply(Yt).operate(z2);
		}
		
		return toParameter(alpha, beta);
	}

	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		
		//Calculating X statistic
		List<double[]> currentStatistic1 = Util.newList();
		for (int i = 0; i < this.xData.size(); i++) {
			double[] profile = this.xData.get(i);
			if (!Util.isUsed(profile[profile.length - 1])) continue;

			double[] newProfile = new double[profile.length];
			for (int j = 0; j < profile.length; j++) newProfile[j] = profile[j];
			currentStatistic1.add(newProfile);
		}
		
		//Calculating Y statistic
		List<double[]> currentStatistic2 = Util.newList();
		for (int i = 0; i < this.yData.size(); i++) {
			double[] profile = this.yData.get(i);
			if (!Util.isUsed(profile[profile.length - 1])) continue;
			
			double[] newProfile = new double[profile.length];
			for (int j = 0; j < profile.length; j++) newProfile[j] = profile[j];
			currentStatistic2.add(newProfile);
		}

		try {
			return (double[]) maximization(new Object[] {currentStatistic1, currentStatistic2});
		}
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		double[] parameter0 = new double[this.n + this.k + 2];
		int N = this.n + this.k + 2;
		for (int i = 0; i < N; i++) parameter0[i] = 0;
		return parameter0;
	}

	
	@Override
	protected boolean terminatedCondition(Object currentParameter, Object estimatedParameter, Object... info) {
		// TODO Auto-generated method stub
		double[] parameter1 = (double[])currentParameter;
		double[] parameter2 = (double[])estimatedParameter;
		double maxBias = 0;
		for (int i = 0; i < parameter1.length; i++) {
			double bias = Math.abs(parameter1[i] - parameter2[i]);
			if (maxBias < bias)
				maxBias = bias;
		}
		
		return maxBias <= getConfig().getAsReal(EM_EPSILON_FIELD);
	}

	
	/**
	 * Creating compound parameter from two vectors of coefficients.
	 * @param alpha the vector of coefficients of the first regression model. 
	 * @param beta the vector of coefficients of the second regression model.
	 * @return compound parameter from two vectors of coefficients of two regression models.
	 */
	private double[] toParameter(RealVector alpha, RealVector beta) {
		int size1 = alpha.getDimension();
		int size2 = beta.getDimension();
		double[] parameter = new double[size1 + size2];
		for (int i = 0; i < size1; i++)
			parameter[i] = alpha.getEntry(i);
		for (int i = 0; i < size2; i++)
			parameter[i + size1] = beta.getEntry(i);
		
		return parameter;
	}
	
	
	/**
	 * Extracting vector of coefficients of the first regression model from the specified parameter.
	 * @param parameter specified parameter.
	 * @return vector of coefficients of the first regression model.
	 */
	private RealVector extractAlpha(double[] parameter) {
		int size = n + 1;
		RealVector alpha = new ArrayRealVector(size);
		for (int i = 0; i < size; i++)
			alpha.setEntry(i, parameter[i]);
		return alpha;
	}
	
	
	/**
	 * Extracting vector of coefficients of the second regression model from the specified parameter.
	 * @param parameter specified parameter.
	 * @return vector of coefficients of the second regression model.
	 */
	private RealVector extractBeta(double[] parameter) {
		int size = k + 1;
		int start = n + 1;
		RealVector beta = new ArrayRealVector(size);
		for (int i = 0; i < size; i++)
			beta.setEntry(i, parameter[i + start]);
		return beta;
	}
	
	
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		double[] dualCoeffs = (double[])estimatedParameter;
		if (dualCoeffs == null || dualCoeffs.length == 0)
			return Constants.UNUSED;
		
		if (input == null || !(input instanceof Profile))
			return null; //only support profile input currently
		Profile profile = (Profile)input;
		
		boolean executionMode = getConfig().getAsBoolean(EXECUTION_FIRST_MODE_FIELD);
		if (executionMode) {
			RealVector alpha = extractAlpha(dualCoeffs);
			return execute(alpha, 0, profile);
		}
		else {
			RealVector beta = extractBeta(dualCoeffs);
			return execute(beta, n, profile);
		}
	}


	/**
	 * Executing regression model with specified coefficients and parameter.
	 * @param coeffs specified coefficients.
	 * @param start starting index
	 * @param profile specified profile.
	 * @return result of execution of regression model with specified coefficients and parameter.
	 */
	private static double execute(RealVector coeffs, int start, Profile profile) {
		if (coeffs.getDimension() == 0)
			return Constants.UNUSED;
		
		double sum = coeffs.getEntry(0);
		int count = coeffs.getDimension() - 1;
		for (int i = 0; i < count; i++) {
			double value = profile.getValueAsReal(start + i);
			sum += coeffs.getEntry(i + 1) * value; 
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
			return "drem";
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		DualRegressionEM em = new DualRegressionEM();
		em.getConfig().putAll((DataConfig)this.getConfig().clone());
		return em;
	}


	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(DREM_DUAL_FIELD, DREM_DUAL_DEFAULT); // dual model is true.
		config.put(DREM_X_NUMBER_FIELD, DREM_X_NUMBER_DEFAULT); // The number of X variables
		config.put(DREM_Y_NUMBER_FIELD, DREM_Y_NUMBER_DEFAULT); // The number of Y variables
		config.put(EXECUTION_FIRST_MODE_FIELD, EXECUTION_FIRST_MODE_DEFAULT); // execution mode
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}


	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		double[] dualCoeffs = (double[])estimatedParameter;
		if (dualCoeffs == null || dualCoeffs.length == 0)
			return "";

		StringBuffer buffer = new StringBuffer();
		RealVector alpha = extractAlpha(dualCoeffs);
		buffer.append(attList.get(attList.size() - 1).getName() + " = " + alpha.getEntry(0));
		for (int i = 0; i < n; i++) {
			double coeff = alpha.getEntry(i + 1);
			if (coeff < 0)
				buffer.append(" - " + Math.abs(coeff) + "*" + attList.get(i).getName());
			else
				buffer.append(" + " + coeff + "*" + attList.get(i).getName());
		}
		
		buffer.append(", ");
		
		RealVector beta = extractBeta(dualCoeffs);
		buffer.append(attList.get(attList.size() - 1).getName() + " = " + beta.getEntry(0));
		for (int i = 0; i < k; i++) {
			double coeff = beta.getEntry(i + 1);
			if (coeff < 0)
				buffer.append(" - " + Math.abs(coeff) + "*" + attList.get(i + n).getName());
			else
				buffer.append(" + " + coeff + "*" + attList.get(i + n).getName());
		}
		
		return buffer.toString();
	}


	@Override
	public String parameterToShownText(Object parameter) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof double[]))
			return "";
		double[] array = (double[])parameter;
		return TextParserUtil.toText(array, ",");
	}

	
}
