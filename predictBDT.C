//Charlie Mueller 2/24/2016
#include <iostream>
#include <fstream>
#include "TSystem.h"
#include <vector>
#include "TH1.h"
#include "TChain.h"
#include <string>
#include <algorithm>
#include "TString.h"
#include "TH1D.h"
#include "TFile.h"
#include <cmath>
#include "TLorentzVector.h"
#include "ttH-13TeVMultiLeptons/TemplateMakers/src/classes.h"
#include "TMVA/Config.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#include "selection.h"
#include "loadSamples.h"
#include "treeTools.h"
#include "TCanvas.h"
#include "TH2I.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <dirent.h>

/////////////////////////////////////////
///
/// usage: root -l makeSelectionTrees.C+
///
/////////////////////////////////////////


// Sorts from high to low b-jet probability
struct reverse_probb_key
{
    inline bool operator() (const ttH::Jet& struct1, const ttH::Jet& struct2)
    {
        return ((struct1.DeepCSVprobb + struct1.DeepCSVprobbb) < (struct2.DeepCSVprobb + struct2.DeepCSVprobbb));
    }
};

struct pt_key
{
    inline bool operator() (const ttH::Jet& struct1, const ttH::Jet& struct2)
    {
        return (struct1.obj.pt() > struct2.obj.pt());
    }
};

void run_it(TChain* tree, TString prediction_filename)
{

  int treeentries = tree->GetEntries();   
  cout << "# events in tree: "<< treeentries << endl;

  double mcwgt_intree = -999.;
  double wallTimePerEvent_intree = -99.;  
  int eventnum_intree = -999;
  int lumiBlock_intree = -999;
  int runNumber_intree = -999;
  
  Int_t *class = 0;
  ttH::Jet *tree_jet1 = 0;
  ttH::Jet *tree_jet2 = 0;
  ttH::Jet *tree_jet3 = 0;
  vector<ttH::Jet> *preselected_jets_intree=0;
  
  tree->SetBranchStatus("*",1);

  tree->SetBranchAddress("class", &class);
  tree->SetBranchAddress("jet1", &tree_jet1);
  tree->SetBranchAddress("jet2", &tree_jet2);
  tree->SetBranchAddress("jet3", &tree_jet3);
    
  float b_pt = -99;
  float b_mass = -99;
  float b_ptD = -99;
  float b_axis1 = -99;
  float b_mult = -99;
  float b_csv = -99;
  float b_cvsb = -99;
  float b_cvsl = -99;

  float wj1_pt = -99;
  float wj1_mass = -99;
  float wj1_ptD = -99;
  float wj1_axis1 = -99;
  float wj1_mult = -99;
  float wj1_csv = -99;
  float wj1_cvsb = -99;
  float wj1_cvsl = -99;

  float wj2_pt = -99;
  float wj2_mass = -99;
  float wj2_ptD = -99;
  float wj2_axis1 = -99;
  float wj2_mult = -99;
  float wj2_csv = -99;
  float wj2_cvsb = -99;
  float wj2_cvsl = -99;

  float b_wj1_deltaR = -99;
  float b_wj1_mass = -99;
  float b_wj2_deltaR = -99;
  float b_wj2_mass = -99;
  float w_deltaR = -99;
  float w_mass = -99;
  float b_w_deltaR = -99;
  float top_mass = -99;
    
  TMVA::Reader* reader = new TMVA::Reader( "!Color:!Silent" );

  reader->AddVariable("var_b_pt",&b_pt);
  reader->AddVariable("var_b_mass",&b_mass);
  reader->AddVariable("var_b_ptD",&b_ptD);
  reader->AddVariable("var_b_axis1",&b_axis1);
  reader->AddVariable("var_b_mult",&b_mult);
  reader->AddVariable("var_b_deepcsv_bvsall",&b_csv);
  reader->AddVariable("var_b_deepcsv_cvsb",&b_cvsb);
  reader->AddVariable("var_b_deepcsv_cvsl",&b_cvsl);

  reader->AddVariable("var_wj1_pt",&wj1_pt);
  reader->AddVariable("var_wj1_mass",&wj1_mass);
  reader->AddVariable("var_wj1_ptD",&wj1_ptD);
  reader->AddVariable("var_wj1_axis1",&wj1_axis1);
  reader->AddVariable("var_wj1_mult",&wj1_mult);
  reader->AddVariable("var_wj1_deepcsv_bvsall",&wj1_csv);
  reader->AddVariable("var_wj1_deepcsv_cvsb",&wj1_cvsb);
  reader->AddVariable("var_wj1_deepcsv_cvsl",&wj1_cvsl);

  reader->AddVariable("var_wj2_pt",&wj2_pt);
  reader->AddVariable("var_wj2_mass",&wj2_mass);
  reader->AddVariable("var_wj2_ptD",&wj2_ptD);
  reader->AddVariable("var_wj2_axis1",&wj2_axis1);
  reader->AddVariable("var_wj2_mult",&wj2_mult);
  reader->AddVariable("var_wj2_deepcsv_bvsall",&wj2_csv);
  reader->AddVariable("var_wj2_deepcsv_cvsb",&wj2_cvsb);
  reader->AddVariable("var_wj2_deepcsv_cvsl",&wj2_cvsl);

  reader->AddVariable("var_b_wj1_deltaR",&b_wj1_deltaR);
  reader->AddVariable("var_b_wj1_mass",&b_wj1_mass);
  reader->AddVariable("var_b_wj2_deltaR",&b_wj2_deltaR);
  reader->AddVariable("var_b_wj2_mass",&b_wj2_mass);
  reader->AddVariable("var_wcand_deltaR",&w_deltaR);
  reader->AddVariable("var_wcand_mass",&w_mass);
  reader->AddVariable("var_b_wcand_deltaR",&b_w_deltaR);
  reader->AddVariable("var_topcand_mass",&top_mass);
    
  reader->BookMVA("BDT", "/afs/crc.nd.edu/user/m/mdrnevic/Private/resTop_xgb_csv_order_deepCTag.xml");
  
  Int_t cachesize = 500000000;   //500 MBytes
  tree->SetCacheSize(cachesize);
  tree->SetCacheLearnEntries(20); 
    
  ofstream prediction_file;
  prediction_file.open(prediction_filename);

  double starttime = get_wall_time();
  for (int i=0; i<treeentries; i++)
  {

        tree->GetEntry(i);
        cout<< "Finished: " <<(i+1)*100/treeentries <<"%\r";
        vector<ttH::Jet> jet_triplet = {*tree_jet1, *tree_jet2, *tree_jet3};
        sort(jet_triplet.begin(), jet_triplet.end(), reverse_probb_key());
        ttH::Jet bjet = jet_triplet.back();
        jet_triplet.pop_back();
        sort(jet_triplet.begin(), jet_triplet.end(), pt_key());
        ttH::Jet wj1 = jet_triplet[0];
        ttH::Jet wj2 = jet_triplet[1];
            
        b_pt = bjet.obj.pt();
        b_mass = bjet.obj.M();
        b_csv = bjet.DeepCSVprobb + bjet.DeepCSVprobbb;
        b_cvsl = bjet.DeepCSVprobc/(bjet.DeepCSVprobc + bjet.DeepCSVprobudsg);
        b_cvsb = bjet.DeepCSVprobc/(bjet.DeepCSVprobc + bjet.DeepCSVprobb + bjet.DeepCSVprobbb);
        b_ptD = bjet.ptD;
        b_axis1 = bjet.axis1;
        b_mult = bjet.mult;

        wj1_pt = wj1.obj.pt();
        wj1_mass = wj1.obj.M();
        wj1_csv = wj1.DeepCSVprobb + wj1.DeepCSVprobbb;
        wj1_cvsl = wj1.DeepCSVprobc/(wj1.DeepCSVprobc + wj1.DeepCSVprobudsg);
        wj1_cvsb = wj1.DeepCSVprobc/(wj1.DeepCSVprobc + wj1.DeepCSVprobb + wj1.DeepCSVprobbb);
        wj1_ptD = wj1.ptD;
        wj1_axis1 = wj1.axis1;
        wj1_mult = wj1.mult;

        wj2_pt = wj2.obj.pt();
        wj2_mass = wj2.obj.M();
        wj2_csv = wj2.DeepCSVprobb + wj2.DeepCSVprobbb;
        wj2_cvsl = wj2.DeepCSVprobc/(wj2.DeepCSVprobc + wj2.DeepCSVprobudsg);
        wj2_cvsb = wj2.DeepCSVprobc/(wj2.DeepCSVprobc + wj2.DeepCSVprobb + wj2.DeepCSVprobbb);
        wj2_ptD = wj2.ptD;
        wj2_axis1 = wj2.axis1;
        wj2_mult = wj2.mult;

        TLorentzVector *tvec1 = new TLorentzVector();
        TLorentzVector *tvec2 = new TLorentzVector();
        TLorentzVector *tvec3 = new TLorentzVector();
        tvec1->SetPtEtaPhiM(bjet.obj.pt(), bjet.obj.eta(), bjet.obj.phi(), bjet.obj.M());
        tvec2->SetPtEtaPhiM(wj1.obj.pt(), wj1.obj.eta(), wj1.obj.phi(), wj1.obj.M());
        tvec3->SetPtEtaPhiM(wj2.obj.pt(), wj2.obj.eta(), wj2.obj.phi(), wj2.obj.M());
        TLorentzVector W = *tvec2 + *tvec3;
        TLorentzVector top = *tvec1 + *tvec2 + *tvec3;

        b_wj1_deltaR = deltaR(tvec1->Eta(), tvec1->Phi(), tvec2->Eta(), tvec2->Phi());
        b_wj1_mass = (*tvec1 + *tvec2).M();
        b_wj2_deltaR = deltaR(tvec1->Eta(), tvec1->Phi(), tvec3->Eta(), tvec3->Phi());
        b_wj2_mass =(*tvec1 + *tvec3).M();
        w_deltaR = deltaR(tvec2->Eta(), tvec2->Phi(), tvec3->Eta(), tvec3->Phi());
        w_mass = W.M();
        b_w_deltaR = deltaR(tvec1->Eta(), tvec1->Phi(), W.Eta(), W.Phi());
        top_mass = top.M();
              
        float score = reader->EvaluateMVA("BDT");
      
        prediction_file << *class <<","<< score <<"\n";
        prediction_file.flush();
  }

  prediction_file.close()
  double endtime = get_wall_time();
  cout << "Elapsed time: " << endtime - starttime << " seconds, " << endl;
  if (treeentries>0) cout << "an average of " << (endtime - starttime) / treeentries << " per event." << endl;
}

void predictBDT(TString sample="")
{  
  TChain *tth_chain = new TChain("tripleTree");    
  tth_chain->Add("/scratch365/mdrnevic/trees/testing/ttH/testing_DeepLearningTree.root");
  TString prediction_filename = "testing_BDT_score.csv";
  run_it(tth_chain, prediction_filename);
}
