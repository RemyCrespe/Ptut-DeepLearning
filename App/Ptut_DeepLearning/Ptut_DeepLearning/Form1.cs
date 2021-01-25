using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;

namespace Ptut_DeepLearning
{
    public partial class Ctrl_Ptut : Form
    {
        private string Path_c;
        private string BrainPath_c;
        private string MaskPath_c;
        public Ctrl_Ptut()
        {
            InitializeComponent();
        }

        private void Btn_ChargerImage_Click(object sender, EventArgs e)
        {
            DialogResult dr = this.openFileDialog1.ShowDialog();
            if (dr == DialogResult.OK)
            {
                string path = this.openFileDialog1.FileName;
                Path_c = path;
                Img_Entree.SizeMode = PictureBoxSizeMode.StretchImage;
                Img_Entree.Image = new Bitmap(this.openFileDialog1.FileName);
            }
        }

        private void run_cmd(string cmd, string args)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = "python";
            start.Arguments = string.Format("{0} {1}", cmd, args);
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;

            using (Process process = Process.Start(start))
            {
                process.Exited += (sender, e) =>
                {
                    using (StreamReader reader = process.StandardOutput)
                    {
                        string result = reader.ReadToEnd();
                        //Console.Write(result);

                        MessageBox.Show(result);
                    }
                };
                process.WaitForExit();
            }
        }

        private void Btn_Predict_Click(object sender, EventArgs e)
        {
            run_cmd("D:/ptut/Ptut-DeepLearning/Prediction.py", Path_c);
            Img_Sortie.SizeMode = PictureBoxSizeMode.StretchImage;
            Img_Sortie.Image = new Bitmap("temp.jpg");
        }

        private void Check_mask_CheckStateChanged(object sender, EventArgs e)
        {
            if (Check_mask.Checked)
            {
                pnl_mask.Visible = true;
            }
            else
            {
                pnl_mask.Visible = false;
            }

        }

        private void Btn_mask_Click(object sender, EventArgs e)
        {
            DialogResult dr = this.openFileDialog1.ShowDialog();
            if (dr == DialogResult.OK)
            {
                string path = this.openFileDialog1.FileName;
                img_maskEntree.SizeMode = PictureBoxSizeMode.StretchImage;
                img_maskEntree.Image = new Bitmap(this.openFileDialog1.FileName);
            }
        }

        private void Btn_Modele_Click(object sender, EventArgs e)
        {
            DialogResult dr = this.openFileDialog1.ShowDialog();
            if (dr == DialogResult.OK)
            {
                string path = this.openFileDialog1.FileName;
                //System.IO.File.Copy(path, System.IO.Path.GetDirectoryName(Application.ExecutablePath));
                string direct_path = System.IO.Path.GetDirectoryName(Application.StartupPath) + "\\Debug\\model.h5";
                System.IO.File.Copy(path, direct_path, true);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            DialogResult result = folderBrowserDialog1.ShowDialog();
            if (result == DialogResult.OK)
            {
                string folderName = folderBrowserDialog1.SelectedPath + '\\';
                BrainPath_c = folderName;
                BrainPath_text.Text = folderName;

            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            DialogResult result = folderBrowserDialog1.ShowDialog();
            if (result == DialogResult.OK)
            {
                string folderName = folderBrowserDialog1.SelectedPath + '\\';
                //MessageBox.Show(folderName);
                MaskPath_c = folderName;
                MaskPath_text.Text = folderName;
                //openMenuItem.PerformClick();

            }

        }

        private void button3_Click(object sender, EventArgs e)
        {
            string arg_l;

            arg_l = BrainPath_c + ' ' + MaskPath_c + ' '+ Text_batch.Text + ' ' + Text_TrainStep.Text + ' ' + Text_ValStep.Text + ' ' + Text_Epoch.Text; 
            
            run_cmd("D:/ptut/Ptut-DeepLearning/Entrainement.py", arg_l);
            MessageBox.Show("Le modèle est entrainé");
        }
    }
}
