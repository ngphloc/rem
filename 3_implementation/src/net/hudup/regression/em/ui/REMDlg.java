package net.hudup.regression.em.ui;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.awt.print.PrinterJob;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableModel;

import net.hudup.core.Util;
import net.hudup.core.data.Attribute;
import net.hudup.core.data.AttributeList;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.logistic.ui.UIUtil;
import net.hudup.regression.AbstractRM;
import net.hudup.regression.em.REMImpl;
import net.hudup.regression.em.ui.graph.Graph;
import net.hudup.regression.em.ui.graph.PlotGraphExt;

/**
 * This class represents the dialog to show content of regression model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class REMDlg extends JDialog {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal regression model.
	 */
	protected REMImpl rem = null;
	
	
	/**
	 * Regression table.
	 */
	private JTable tblRegression = null;
	
	
	/**
	 * Calculation text.
	 */
	private JTextField txtCalc = null; 
	
	
	/**
	 * Constructor with specified regression model.
	 * @param comp parent component.
	 * @param rem specified regression model.
	 */
	public REMDlg(final Component comp, final REMImpl rem) {
		super(UIUtil.getFrameForComponent(comp), "Regression Information", true);
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		setSize(600, 400);
		setLocationRelativeTo(UIUtil.getFrameForComponent(comp));
		
		this.rem = rem;
		
		setLayout(new BorderLayout());
		
		JPanel top = new JPanel(new BorderLayout());
		this.add(top, BorderLayout.NORTH);
		
		REMTextArea txtModel = new REMTextArea(rem);
		top.add(txtModel, BorderLayout.CENTER);
		
		JPanel main = new JPanel(new BorderLayout());
		this.add(main, BorderLayout.CENTER);
		
		
		//Header of main panel
		JPanel header = new JPanel(new GridLayout(1, 0));
		main.add(header, BorderLayout.NORTH);

		JPanel col = null;
		JPanel left = null;
		JPanel right = null; 
		JPanel pane = null;
		
		col = new JPanel(new BorderLayout());
		header.add(col);
		//
		left = new JPanel(new GridLayout(0, 1));
		col.add(left, BorderLayout.WEST);
		//
		left.add(new JLabel("Variance: "));
		left.add(new JLabel("R: "));
		left.add(new JLabel("Ratio error: "));
		//
		right = new JPanel(new GridLayout(0, 1));
		col.add(right, BorderLayout.CENTER);
		//
		double variance = rem.getExchangedParameter().getZVariance();
		variance = Util.isUsed(variance) ? variance : rem.getExchangedParameter().estimateZVariance(rem.getLargeStatistics()); 
		JTextField txtVariance = new JTextField(MathUtil.format(variance));
		txtVariance.setCaretPosition(0);
		txtVariance.setEditable(false);
		pane = new JPanel(new BorderLayout());
		pane.add(txtVariance, BorderLayout.WEST);
		right.add(pane);
		//
		JTextField txtR = new JTextField(
				MathUtil.format(-1));
		txtR.setCaretPosition(0);
		txtR.setEditable(false);
		pane = new JPanel(new BorderLayout());
		pane.add(txtR, BorderLayout.WEST);
		right.add(pane);
		//
		JTextField txtError = new JTextField(
				MathUtil.format(-1));
		txtError.setCaretPosition(0);
		txtError.setEditable(false);
		pane = new JPanel(new BorderLayout());
		pane.add(txtError, BorderLayout.WEST);
		right.add(pane);
		
		
		//Body of main panel
		JPanel body = new JPanel(new BorderLayout());
		main.add(body, BorderLayout.CENTER);
		
		JPanel paneRegressors = new JPanel(new BorderLayout());
		body.add(paneRegressors, BorderLayout.NORTH);
		
		List<RegressorWrapper> regressorList = RegressorWrapper.getRegressorList(rem);
		regressorList.sort(new Comparator<RegressorWrapper>() {

			@Override
			public int compare(RegressorWrapper o1, RegressorWrapper o2) {
				// TODO Auto-generated method stub
				return o1.getName().compareToIgnoreCase(o2.getName());
			}
			
		});
		JComboBox<RegressorWrapper> cmbReggressors = new JComboBox<RegressorWrapper>(regressorList.toArray(new RegressorWrapper[] {}));
		paneRegressors.add(cmbReggressors, BorderLayout.CENTER);
		JButton btnPlot = new JButton(new AbstractAction("Plot") {

			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public void actionPerformed(ActionEvent e) {
				RegressorWrapper regressor = (RegressorWrapper)cmbReggressors.getSelectedItem();
				if (regressor == null)
					JOptionPane.showMessageDialog(
							cmbReggressors, 
							"No selected regressor", 
							"No selected regressor", 
							JOptionPane.ERROR_MESSAGE);
				else
					plot2dDecomposedGraph(regressor.getIndex());
			}
		});
		paneRegressors.add(btnPlot, BorderLayout.EAST);
		
		
		//Footer of main panel
		JPanel footer = new JPanel(new BorderLayout());
		main.add(footer, BorderLayout.SOUTH);
		
		JPanel control = new JPanel();
		footer.add(control, BorderLayout.CENTER);
		
		DefaultTableModel tbm = new DefaultTableModel() {

			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public boolean isCellEditable(int row, int column) {
				return (column == 1);
			}
		};
		tbm.setColumnIdentifiers(new String[] {"Regressor", "Value"});
		for (RegressorWrapper regressor : regressorList) {
			Vector<Object> rowData = new Vector<Object>();
			rowData.add(regressor);
			rowData.add(new Double(0));
			tbm.addRow(rowData);
		}
		this.tblRegression = new JTable(tbm);
		this.tblRegression.setPreferredScrollableViewportSize(new Dimension(200, 60));   
		JScrollPane scroll = new JScrollPane(this.tblRegression);
		control.add(scroll);
		
	    JButton btnCalc = new JButton("Calculate");
	    btnCalc.addActionListener(new ActionListener() {
	    	public void actionPerformed(ActionEvent arg0) {
	    		calc();
	    	}
	    });
		control.add(btnCalc);

		this.txtCalc = new JTextField(12);
		txtCalc.setEditable(false);
		control.add(txtCalc);
		
		setVisible(true);
	}
	
	
	/**
	 * Calculating regression model.
	 */
	private void calc() {
		if (rem == null) {
			JOptionPane.showMessageDialog(
					this, 
					"Null regression model", 
					"Null regression model", 
					JOptionPane.ERROR_MESSAGE);
			return;
		}

		try {
			Map<String, Double> regressorValues = Util.newMap();
			int n = tblRegression.getModel().getRowCount();
			TableModel tbm = tblRegression.getModel();
			for (int i = 0; i < n; i++) {
				String name = tbm.getValueAt(i, 0).toString();
				String value = tbm.getValueAt(i, 1).toString();
				
				regressorValues.put(name, Double.parseDouble(value));
			}
			
			double value = AbstractRM.extractNumber(rem.execute(regressorValues));
			if (!Util.isUsed(value)) {
				JOptionPane.showMessageDialog(
						this, 
						"Regression model is not executed", 
						"Failed execution", 
						JOptionPane.ERROR_MESSAGE);
				this.txtCalc.setText("");
			}
			else {
				this.txtCalc.setText(MathUtil.format(value));
				this.txtCalc.setCaretPosition(0);
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			this.txtCalc.setText("");
		}
			
	}

	
	/**
	 * Plotting the graph of given regressor.
	 * @param regressorIndex
	 */
	private void plot2dDecomposedGraph(int regressorIndex) {
		Graph graph = rem != null ? rem.create2dDecomposedGraph(regressorIndex) : null;
		if (graph == null) {
			JOptionPane.showMessageDialog(
					this, 
					"Cannot create graph", 
					"Cannot create graph", 
					JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		final JDialog dlg = new JDialog(UIUtil.getFrameForComponent(this), "Graph", true);
		dlg.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		dlg.setSize(600, 400);
		dlg.setLocationRelativeTo(UIUtil.getFrameForComponent(this));
		
		dlg.setLayout(new BorderLayout());
		dlg.add( (Component)graph, BorderLayout.CENTER);
		
		JPanel footer = new JPanel();
		dlg.add(footer, BorderLayout.SOUTH);
		
		JButton btnExport = new JButton("Export image");
		btnExport.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				graph.exportImage();
			}
		});
		footer.add(btnExport);
		
		JButton btnPrint = new JButton("Print");
		btnPrint.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				try {
					Printable printable = new Printable() {
						
						@Override
						public int print(Graphics graphics, PageFormat pageFormat, int pageIndex)
								throws PrinterException {
							// TODO Auto-generated method stub
							if (pageIndex > 0)
								return NO_SUCH_PAGE;
							
							double x = pageFormat.getImageableX();
							double y = pageFormat.getImageableY();
							graphics.translate((int)x, (int)y);
							((PlotGraphExt) graph).paint(graphics, (int) pageFormat.getImageableWidth(), (int) pageFormat.getImageableHeight());
							
							return PAGE_EXISTS;
						}
					};
					
	  				PrinterJob pjob = PrinterJob.getPrinterJob();
	  				
	    			//set a HelloPrint as the target to print
	  				pjob.setPrintable(printable);
	  				
	  				//get the print dialog, continue if canel
	  				//is not clicked
    				if (pjob.printDialog()) {
    					//print the target (HelloPrint)
    					pjob.print();
    				}
    				
				}
				catch (Throwable ex) {
					ex.printStackTrace();
				}
			}
		});
		footer.add(btnPrint);
		
		JButton btnClose = new JButton("Close");
		btnClose.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				dlg.dispose();
			}
		});
		footer.add(btnClose);

		dlg.setVisible(true);
	}
	
	
	/**
	 * This class represents the wrapper of a regressor.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	protected static class RegressorWrapper {
		
		/**
		 * Regressor name.
		 */
		protected String name = null;
		
		/**
		 * Regressor index.
		 */
		protected int index = -1;
		
		/**
		 * Constructor with specified name and index.
		 * @param name specified name.
		 * @param index specified index.
		 */
		public RegressorWrapper(String name, int index) {
			this.name = name;
			this.index = index;
		}
		
		/**
		 * Getting name.
		 * @return name.
		 */
		public String getName() {
			return name;
		}
		
		/**
		 * Getting index.
		 * @return index.
		 */
		public int getIndex() {
			return index;
		}

		@Override
		public String toString() {
			// TODO Auto-generated method stub
			return name;
		}
		
		/**
		 * Getting list of regressors from specified regression model.
		 * @param rem specified regression model.
		 * @return list of regressors from specified regression model.
		 */
		public static List<RegressorWrapper> getRegressorList(REMImpl rem){
			List<RegressorWrapper> wrapperList = Util.newList();
			if (rem == null || rem.getAttributeList() == null)
				return wrapperList;
			AttributeList attList = rem.getAttributeList();
			for (int i = 0; i < attList.size(); i++) {
				Attribute att = attList.get(i);
				RegressorWrapper wrapper = new RegressorWrapper(att.getName(), att.getIndex() + 1);
				wrapperList.add(wrapper);
			}
			return wrapperList;
		}
		
	}
}
