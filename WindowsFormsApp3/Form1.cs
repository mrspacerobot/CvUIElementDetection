using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Flann;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowsFormsApp3
{
    public partial class Form1 : Form
    {
        
        public Form1()
        {
            InitializeComponent();
        }
       

        private void detectToolStripMenuItem_Click(object sender, EventArgs e)
        {
            using (Mat modelImage = CvInvoke.Imread("C:\\Users\\dylan\\Desktop\\19_FindWpfByAutomationId_Button.jpg", ImreadModes.AnyColor))
            using (Mat observedImage = CvInvoke.Imread("C:\\Users\\dylan\\Desktop\\Capture.jpg", ImreadModes.AnyColor))
            {
                Mat result = DrawMatches.Draw(modelImage,observedImage);
                //Mat result = DrawMatches.ProcessImage(modelImage);
                imageBox2.Image = result;
            }





            
                    
                
            
        }

        private void tableLayoutPanel1_Paint(object sender, PaintEventArgs e)
        {

        }
    }
}
