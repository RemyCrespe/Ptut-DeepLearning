namespace Ptut_DeepLearning
{
    partial class Ctrl_Ptut
    {
        /// <summary>
        /// Variable nécessaire au concepteur.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Nettoyage des ressources utilisées.
        /// </summary>
        /// <param name="disposing">true si les ressources managées doivent être supprimées ; sinon, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Code généré par le Concepteur Windows Form

        /// <summary>
        /// Méthode requise pour la prise en charge du concepteur - ne modifiez pas
        /// le contenu de cette méthode avec l'éditeur de code.
        /// </summary>
        private void InitializeComponent()
        {
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.button3 = new System.Windows.Forms.Button();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.Text_batch = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.MaskPath_text = new System.Windows.Forms.TextBox();
            this.button2 = new System.Windows.Forms.Button();
            this.BrainPath_text = new System.Windows.Forms.TextBox();
            this.button1 = new System.Windows.Forms.Button();
            this.Text_Epoch = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.Text_ValStep = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.Text_TrainStep = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.Btn_Modele = new System.Windows.Forms.Button();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.Check_mask = new System.Windows.Forms.CheckBox();
            this.pnl_mask = new System.Windows.Forms.Panel();
            this.Btn_mask = new System.Windows.Forms.Button();
            this.img_maskEntree = new System.Windows.Forms.PictureBox();
            this.Img_Sortie = new System.Windows.Forms.PictureBox();
            this.Btn_Predict = new System.Windows.Forms.Button();
            this.Btn_ChargerImage = new System.Windows.Forms.Button();
            this.Img_Entree = new System.Windows.Forms.PictureBox();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.groupBox1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.pnl_mask.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.img_maskEntree)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Img_Sortie)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Img_Entree)).BeginInit();
            this.SuspendLayout();
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl1.Location = new System.Drawing.Point(0, 0);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(1227, 465);
            this.tabControl1.TabIndex = 0;
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.button3);
            this.tabPage1.Controls.Add(this.groupBox1);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(1219, 439);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Entrainement Modele";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(781, 117);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(254, 155);
            this.button3.TabIndex = 1;
            this.button3.Text = "Lancer l\'entrainement";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.Text_batch);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.MaskPath_text);
            this.groupBox1.Controls.Add(this.button2);
            this.groupBox1.Controls.Add(this.BrainPath_text);
            this.groupBox1.Controls.Add(this.button1);
            this.groupBox1.Controls.Add(this.Text_Epoch);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.Text_ValStep);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.Text_TrainStep);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Location = new System.Drawing.Point(45, 25);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(553, 319);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Paramètres d\'entrainement du modèle";
            // 
            // Text_batch
            // 
            this.Text_batch.Location = new System.Drawing.Point(197, 133);
            this.Text_batch.Name = "Text_batch";
            this.Text_batch.Size = new System.Drawing.Size(109, 20);
            this.Text_batch.TabIndex = 2;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(27, 136);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(125, 13);
            this.label4.TabIndex = 10;
            this.label4.Text = "Nombres d\'échantillons : ";
            // 
            // MaskPath_text
            // 
            this.MaskPath_text.Location = new System.Drawing.Point(30, 79);
            this.MaskPath_text.Name = "MaskPath_text";
            this.MaskPath_text.Size = new System.Drawing.Size(242, 20);
            this.MaskPath_text.TabIndex = 1;
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(312, 76);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(196, 23);
            this.button2.TabIndex = 8;
            this.button2.Text = "Charger le chemain des masques";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // BrainPath_text
            // 
            this.BrainPath_text.Location = new System.Drawing.Point(30, 38);
            this.BrainPath_text.Name = "BrainPath_text";
            this.BrainPath_text.Size = new System.Drawing.Size(242, 20);
            this.BrainPath_text.TabIndex = 0;
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(312, 36);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(196, 23);
            this.button1.TabIndex = 6;
            this.button1.Text = "Charger le chemain des images";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // Text_Epoch
            // 
            this.Text_Epoch.Location = new System.Drawing.Point(197, 239);
            this.Text_Epoch.Name = "Text_Epoch";
            this.Text_Epoch.Size = new System.Drawing.Size(109, 20);
            this.Text_Epoch.TabIndex = 5;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(18, 242);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(178, 13);
            this.label3.TabIndex = 4;
            this.label3.Text = "Nombres d\'iteration d\'entrainement : ";
            // 
            // Text_ValStep
            // 
            this.Text_ValStep.Location = new System.Drawing.Point(197, 205);
            this.Text_ValStep.Name = "Text_ValStep";
            this.Text_ValStep.Size = new System.Drawing.Size(109, 20);
            this.Text_ValStep.TabIndex = 4;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(18, 208);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(164, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "Nombres d\'étapes de validation : ";
            // 
            // Text_TrainStep
            // 
            this.Text_TrainStep.Location = new System.Drawing.Point(197, 171);
            this.Text_TrainStep.Name = "Text_TrainStep";
            this.Text_TrainStep.Size = new System.Drawing.Size(109, 20);
            this.Text_TrainStep.TabIndex = 3;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(18, 171);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(173, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Nombres d\'étapes d\'entrainement : ";
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.Btn_Modele);
            this.tabPage2.Controls.Add(this.pictureBox1);
            this.tabPage2.Controls.Add(this.Check_mask);
            this.tabPage2.Controls.Add(this.pnl_mask);
            this.tabPage2.Controls.Add(this.Img_Sortie);
            this.tabPage2.Controls.Add(this.Btn_Predict);
            this.tabPage2.Controls.Add(this.Btn_ChargerImage);
            this.tabPage2.Controls.Add(this.Img_Entree);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(1219, 439);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Prediction";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // Btn_Modele
            // 
            this.Btn_Modele.Location = new System.Drawing.Point(216, 14);
            this.Btn_Modele.Name = "Btn_Modele";
            this.Btn_Modele.Size = new System.Drawing.Size(123, 31);
            this.Btn_Modele.TabIndex = 7;
            this.Btn_Modele.Text = "Charger modele";
            this.Btn_Modele.UseVisualStyleBackColor = true;
            this.Btn_Modele.Click += new System.EventHandler(this.Btn_Modele_Click);
            // 
            // pictureBox1
            // 
            this.pictureBox1.Image = global::Ptut_DeepLearning.Properties.Resources.right_arrow;
            this.pictureBox1.Location = new System.Drawing.Point(678, 128);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(107, 105);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox1.TabIndex = 6;
            this.pictureBox1.TabStop = false;
            // 
            // Check_mask
            // 
            this.Check_mask.AutoSize = true;
            this.Check_mask.Location = new System.Drawing.Point(17, 22);
            this.Check_mask.Name = "Check_mask";
            this.Check_mask.Size = new System.Drawing.Size(177, 17);
            this.Check_mask.TabIndex = 5;
            this.Check_mask.Text = "J\'ai un masque avec mon image";
            this.Check_mask.UseVisualStyleBackColor = true;
            this.Check_mask.CheckStateChanged += new System.EventHandler(this.Check_mask_CheckStateChanged);
            // 
            // pnl_mask
            // 
            this.pnl_mask.Controls.Add(this.Btn_mask);
            this.pnl_mask.Controls.Add(this.img_maskEntree);
            this.pnl_mask.Location = new System.Drawing.Point(368, 22);
            this.pnl_mask.Name = "pnl_mask";
            this.pnl_mask.Size = new System.Drawing.Size(241, 351);
            this.pnl_mask.TabIndex = 4;
            this.pnl_mask.Visible = false;
            // 
            // Btn_mask
            // 
            this.Btn_mask.Location = new System.Drawing.Point(40, 264);
            this.Btn_mask.Name = "Btn_mask";
            this.Btn_mask.Size = new System.Drawing.Size(159, 63);
            this.Btn_mask.TabIndex = 2;
            this.Btn_mask.Text = "Charger masque";
            this.Btn_mask.UseVisualStyleBackColor = true;
            this.Btn_mask.Click += new System.EventHandler(this.Btn_mask_Click);
            // 
            // img_maskEntree
            // 
            this.img_maskEntree.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.img_maskEntree.Location = new System.Drawing.Point(20, 30);
            this.img_maskEntree.Name = "img_maskEntree";
            this.img_maskEntree.Size = new System.Drawing.Size(200, 200);
            this.img_maskEntree.TabIndex = 1;
            this.img_maskEntree.TabStop = false;
            // 
            // Img_Sortie
            // 
            this.Img_Sortie.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.Img_Sortie.Location = new System.Drawing.Point(851, 52);
            this.Img_Sortie.Name = "Img_Sortie";
            this.Img_Sortie.Size = new System.Drawing.Size(200, 200);
            this.Img_Sortie.TabIndex = 3;
            this.Img_Sortie.TabStop = false;
            // 
            // Btn_Predict
            // 
            this.Btn_Predict.Location = new System.Drawing.Point(873, 286);
            this.Btn_Predict.Name = "Btn_Predict";
            this.Btn_Predict.Size = new System.Drawing.Size(159, 63);
            this.Btn_Predict.TabIndex = 2;
            this.Btn_Predict.Text = "Prediction";
            this.Btn_Predict.UseVisualStyleBackColor = true;
            this.Btn_Predict.Click += new System.EventHandler(this.Btn_Predict_Click);
            // 
            // Btn_ChargerImage
            // 
            this.Btn_ChargerImage.Location = new System.Drawing.Point(125, 286);
            this.Btn_ChargerImage.Name = "Btn_ChargerImage";
            this.Btn_ChargerImage.Size = new System.Drawing.Size(159, 63);
            this.Btn_ChargerImage.TabIndex = 1;
            this.Btn_ChargerImage.Text = "Charger Image";
            this.Btn_ChargerImage.UseVisualStyleBackColor = true;
            this.Btn_ChargerImage.Click += new System.EventHandler(this.Btn_ChargerImage_Click);
            // 
            // Img_Entree
            // 
            this.Img_Entree.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.Img_Entree.Location = new System.Drawing.Point(105, 52);
            this.Img_Entree.Name = "Img_Entree";
            this.Img_Entree.Size = new System.Drawing.Size(200, 200);
            this.Img_Entree.TabIndex = 0;
            this.Img_Entree.TabStop = false;
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            this.openFileDialog1.Title = "Ouvrir un fichier";
            // 
            // Ctrl_Ptut
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1227, 465);
            this.Controls.Add(this.tabControl1);
            this.Name = "Ctrl_Ptut";
            this.ShowIcon = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Ptut Deep Learning";
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.pnl_mask.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.img_maskEntree)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Img_Sortie)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Img_Entree)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.TabPage tabPage2;
        private System.Windows.Forms.Button Btn_ChargerImage;
        private System.Windows.Forms.PictureBox Img_Entree;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.Button Btn_Predict;
        private System.Windows.Forms.PictureBox Img_Sortie;
        private System.Windows.Forms.Panel pnl_mask;
        private System.Windows.Forms.Button Btn_mask;
        private System.Windows.Forms.PictureBox img_maskEntree;
        private System.Windows.Forms.CheckBox Check_mask;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Button Btn_Modele;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.TextBox Text_Epoch;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox Text_ValStep;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox Text_TrainStep;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private System.Windows.Forms.TextBox MaskPath_text;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.TextBox BrainPath_text;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.TextBox Text_batch;
        private System.Windows.Forms.Label label4;
    }
}

