package net.hudup.regression.em.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.awt.print.PrinterJob;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import javax.imageio.ImageIO;
import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableModel;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.data.DataConfig;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.logistic.UriAssoc;
import net.hudup.core.logistic.xURI;
import net.hudup.core.logistic.ui.UIUtil;
import net.hudup.regression.AbstractRM;
import net.hudup.regression.AbstractRM.VarWrapper;
import net.hudup.regression.RM2;
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
	protected RM2 rm = null;
	
	
	/**
	 * Regression table.
	 */
	private JTable tblRegression = null;
	
	
	/**
	 * Calculation text.
	 */
	private JTextField txtCalc = null; 
	
	
	/**
	 * The first list of graphs.
	 */
	List<Graph> graphList = new ArrayList<Graph>();
	
	
	/**
	 * The second list of graphs.
	 */
	List<Graph> graphList2 = new ArrayList<Graph>();

	
	/**
	 * Constructor with specified regression model.
	 * @param comp parent component.
	 * @param rm specified regression model.
	 */
	public REMDlg(final Component comp, final RM2 rm) {
		super(UIUtil.getFrameForComponent(comp), "Regression Information", true);
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		setSize(800, 600);
		setLocationRelativeTo(UIUtil.getFrameForComponent(comp));
		
		this.rm = rm;
		
		setLayout(new BorderLayout());
		
		JPanel top = new JPanel(new BorderLayout());
		this.add(top, BorderLayout.NORTH);
		
		REMTextArea txtModel = new REMTextArea(rm);
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
		//
		right = new JPanel(new GridLayout(0, 1));
		col.add(right, BorderLayout.CENTER);
		//
		double variance = rm.calcVariance();
		JTextField txtVariance = new JTextField(MathUtil.format(variance));
		txtVariance.setCaretPosition(0);
		txtVariance.setEditable(false);
		pane = new JPanel(new BorderLayout());
		pane.add(txtVariance, BorderLayout.WEST);
		right.add(pane);
		//
		JTextField txtR = new JTextField(
				MathUtil.format(rm.calcR()));
		txtR.setCaretPosition(0);
		txtR.setEditable(false);
		pane = new JPanel(new BorderLayout());
		pane.add(txtR, BorderLayout.WEST);
		right.add(pane);
		
		col = new JPanel(new BorderLayout());
		header.add(col);
		//
		left = new JPanel(new GridLayout(0, 1));
		col.add(left, BorderLayout.WEST);
		//
		left.add(new JLabel("Error mean: "));
		left.add(new JLabel("Error sd: "));
		//
		right = new JPanel(new GridLayout(0, 1));
		col.add(right, BorderLayout.CENTER);
		//
		double[] error = rm.calcError();
		error = (error == null || error.length < 2) ? new double[] {Constants.UNUSED, Constants.UNUSED} : error; 
		JTextField txtRatioErrMean = new JTextField(
				MathUtil.format(error[0]));
		txtRatioErrMean.setCaretPosition(0);
		txtRatioErrMean.setEditable(false);
		pane = new JPanel(new BorderLayout());
		pane.add(txtRatioErrMean, BorderLayout.WEST);
		right.add(pane);
		//
		JTextField txtRatioErrSd = new JTextField(
				MathUtil.format(Math.sqrt(error[1])));
		txtRatioErrSd.setCaretPosition(0);
		txtRatioErrSd.setEditable(false);
		pane = new JPanel(new BorderLayout());
		pane.add(txtRatioErrSd, BorderLayout.WEST);
		right.add(pane);

		
		//Body of main panel
		JPanel body = new JPanel(new BorderLayout());
		main.add(body, BorderLayout.CENTER);
		
		JPanel paneRegressorExprs = new JPanel(new BorderLayout());
		body.add(paneRegressorExprs, BorderLayout.NORTH);
		
		List<VarWrapper> regressorExprs = rm.getRegressorExpressions();
		regressorExprs.sort(new Comparator<VarWrapper>() {

			@Override
			public int compare(VarWrapper o1, VarWrapper o2) {
				// TODO Auto-generated method stub
				return o1.toString().compareToIgnoreCase(o2.toString());
			}
			
		});
		JComboBox<VarWrapper> cmbReggressorExprs = new JComboBox<VarWrapper>(regressorExprs.toArray(new VarWrapper[] {}));
		paneRegressorExprs.add(cmbReggressorExprs, BorderLayout.CENTER);
		JButton btnPlot = new JButton(new AbstractAction("Plot") {

			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public void actionPerformed(ActionEvent e) {
				VarWrapper regressor = (VarWrapper)cmbReggressorExprs.getSelectedItem();
				if (regressor == null)
					JOptionPane.showMessageDialog(
							cmbReggressorExprs, 
							"No selected regressor", 
							"No selected regressor", 
							JOptionPane.ERROR_MESSAGE);
				else
					plotRegressorGraph(regressor.getIndex());
			}
		});
		paneRegressorExprs.add(btnPlot, BorderLayout.EAST);
		
		JPanel paneGraphList = new JPanel(new GridLayout(1, 0));
		body.add(paneGraphList, BorderLayout.CENTER);
		graphList = rm.createResponseRalatedGraphs();
		graphList2 = rm.createResponseRalatedGraphs();
		for (int i = 0; i < graphList.size(); i++) {
			final Graph graph = graphList.get(i);
			
			final Graph graph2 = graphList2.get(i);
			JPanel gPanel = new JPanel(new BorderLayout());
			paneGraphList.add(gPanel);
			
			gPanel.add((Component)graph, BorderLayout.CENTER);
			JPanel toolbar = new JPanel();
			gPanel.add(toolbar, BorderLayout.SOUTH);
			
			JButton btnZoom = UIUtil.makeIconButton("zoomin-16x16.png", 
				"zoom", "Zoom", "Zoom", 
				new ActionListener() {
					
					@Override
					public void actionPerformed(ActionEvent e) {
						final JDialog dlg = new JDialog(UIUtil.getFrameForComponent(comp), "Graph", true);
						dlg.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
						dlg.setSize(600, 400);
						dlg.setLocationRelativeTo(UIUtil.getFrameForComponent(comp));
						
						dlg.setLayout(new BorderLayout());
						dlg.add( (Component)graph2, BorderLayout.CENTER);
						
						JPanel footer = new JPanel();
						dlg.add(footer, BorderLayout.SOUTH);
						JButton btnExport = new JButton("Export image");
						btnExport.addActionListener(new ActionListener() {
							
							@Override
							public void actionPerformed(ActionEvent e) {
								// TODO Auto-generated method stub
								graph2.exportImage();
							}
						});
						footer.add(btnExport);
						
						dlg.setVisible(true);
					}
				});
			btnZoom.setMargin(new Insets(0, 0 , 0, 0));
			toolbar.add(btnZoom);
			
			JButton btnPrint = UIUtil.makeIconButton("print-16x16.png", 
				"print", "Print", "Print", 
				new ActionListener() {
					
					@Override
					public void actionPerformed(ActionEvent e) {
						
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
									((PlotGraphExt) graph2).paint(graphics, (int) pageFormat.getImageableWidth(), (int) pageFormat.getImageableHeight());
									
									return PAGE_EXISTS;
								}
							};
							
			  				PrinterJob pjob = PrinterJob.getPrinterJob();
			  				
			    			//set a HelloPrint as the target to print
			  				pjob.setPrintable(printable);
			  				
			  				//get the print dialog, continue if cancel
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
				btnPrint.setMargin(new Insets(0, 0 , 0, 0));
				toolbar.add(btnPrint);
				
				JButton btnOption = UIUtil.makeIconButton("option-16x16.png", 
						"view_option", "View Option", "View Option", 
						new ActionListener() {
							
							@Override
							public void actionPerformed(ActionEvent e) {
								graph.setupViewOption();
								((PlotGraphExt)graph2).setViewOption(((PlotGraphExt)graph).getViewOption());
							}
						}
				);
				btnOption.setMargin(new Insets(0, 0 , 0, 0));
				toolbar.add(btnOption);
		}

		
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
		List<VarWrapper> regressors = rm.getRegressors();
		regressors.sort(new Comparator<VarWrapper>() {

			@Override
			public int compare(VarWrapper o1, VarWrapper o2) {
				// TODO Auto-generated method stub
				return o1.toString().compareToIgnoreCase(o2.toString());
			}
			
		});
		tbm.setColumnIdentifiers(new String[] {"Regressor", "Value"});
		for (VarWrapper regressor : regressors) {
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
		
		addMouseListener(new MouseAdapter() {

			@Override
			public void mouseClicked(MouseEvent e) {
				// TODO Auto-generated method stub
				if(SwingUtilities.isRightMouseButton(e) ) {

					JPopupMenu contextMenu = createContextMenu();
					if(contextMenu != null) 
						contextMenu.show((Component)e.getSource(), e.getX(), e.getY());
				}
			}
			
		});

		setVisible(true);
	}
	
	
	/**
	 * Plotting the graph of given regressor.
	 * @param xIndex index of given regressor.
	 */
	private void plotRegressorGraph(int xIndex) {
		Graph graph = rm != null ? rm.createRegressorGraph(xIndex) : null;
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
		
		JButton btnOption = new JButton("View option");
		btnOption.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(ActionEvent e) {
					graph.setupViewOption();
				}
			}
		);
		footer.add(btnOption);

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
	 * Calculating regression model.
	 */
	private void calc() {
		if (rm == null) {
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
			
			double value = AbstractRM.extractNumber(rm.execute(regressorValues));
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
	 * Creating context menu.
	 * @return context menu.
	 */
	private JPopupMenu createContextMenu() {
		JPopupMenu contextMenu = new JPopupMenu();
		
		JMenuItem miBigZoom = UIUtil.makeMenuItem(null, "Big zoom", 
			new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					final JDialog dlg = new JDialog(UIUtil.getFrameForComponent(getThis()), "Big zoom", true);
					dlg.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
					dlg.setSize(600, 400);
					dlg.setLocationRelativeTo(UIUtil.getFrameForComponent(getThis()));
					
					dlg.setLayout(new BorderLayout());
					JPanel body = new JPanel(new GridLayout(1, 0));
					dlg.add(body, BorderLayout.CENTER);
					
					for (Graph graph : graphList2) {
						body.add( (Component)graph);
					}
					
					JPanel footer = new JPanel();
					dlg.add(footer, BorderLayout.SOUTH);
					
					JButton btnExport = new JButton("Export image");
					btnExport.addActionListener(new ActionListener() {
						
						@Override
						public void actionPerformed(ActionEvent e) {
							// TODO Auto-generated method stub
							mergeGraphImages(getThis(), graphList2);
						}
					});
					footer.add(btnExport);
					
					dlg.setVisible(true);
				}
			});
		contextMenu.add(miBigZoom);
		
		JMenuItem miExport = UIUtil.makeMenuItem(null, "Export graphs to image", 
			new ActionListener() {
					
				@Override
				public void actionPerformed(ActionEvent e) {
					mergeGraphImages(getThis(), graphList);
				}
			});
		contextMenu.add(miExport);
		
		return contextMenu;
	}
	
	
	/**
	 * Merging list of graphs.
	 * @param comp parent component.
	 * @param graphList list of graphs.
	 */
	private static void mergeGraphImages(Component comp, List<Graph> graphList) {
		if (graphList.size() == 0)
			return;
		
		DataConfig config = new DataConfig();
		config.setStoreUri(xURI.create(new File(".")));
		UriAssoc uriAssoc = Util.getFactory().createUriAssoc(config);
		xURI chooseUri = uriAssoc.chooseUri(comp, false, new String[] {"png"}, new String[] {"PNG file"}, null, "png");
		if (chooseUri == null) {
			JOptionPane.showMessageDialog(
					comp, 
					"Image not exported", 
					"Image not exported", 
					JOptionPane.INFORMATION_MESSAGE);
			return;
		}
		
		int bigWidth = 0;
		int maxHeight = Integer.MIN_VALUE;
		
		for (Graph graph : graphList) {
			bigWidth += graph.getOuterBox().width;
			maxHeight = Math.max(maxHeight, graph.getOuterBox().height);
		}
		
		BufferedImage bigImage = new BufferedImage(bigWidth, maxHeight, BufferedImage.TYPE_INT_ARGB);
		Graphics2D bigGraphics = bigImage.createGraphics();
		
		int x = 0;
		for (Graph graph : graphList) {
			Rectangle outerBox = graph.getOuterBox();
			BufferedImage image = new BufferedImage(outerBox.width, outerBox.height, BufferedImage.TYPE_INT_ARGB);
			Graphics2D graphics = image.createGraphics();
			graphics.setColor(new Color(0, 0, 0));
			
			if (graph instanceof PlotGraphExt)
				((PlotGraphExt)graph).paint(graphics, outerBox.width, outerBox.height);
			else
				graph.paint(graphics);
			
			bigGraphics.drawImage(image, x, 0, null);
			x += outerBox.width;
		}
		
		try {
			ImageIO.write(bigImage, "png", new File(chooseUri.getURI()));
			
			JOptionPane.showMessageDialog(
					comp, 
					"Big image exported successfully", 
					"Big image exported successfully", 
					JOptionPane.INFORMATION_MESSAGE);
		}
		catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	
	/**
	 * Getting this dialog.
	 * @return this dialog.
	 */
	private REMDlg getThis() {
		return this;
	}
	
	
}
