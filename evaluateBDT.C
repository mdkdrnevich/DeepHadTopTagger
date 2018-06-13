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

struct reverse_probb_key
{
    inline bool operator() (const ttH::Jet& struct1, const ttH::Jet& struct2)
    {
        return ((struct1.DeepCSVprobb + struct1.DeepCSVprobbb) > (struct2.DeepCSVprobb + struct2.DeepCSVprobbb));
    }
};

struct pt_key
{
    inline bool operator() (const ttH::Jet& struct1, const ttH::Jet& struct2)
    {
        return (struct1.obj.pt() < struct2.obj.pt());
    }
};

void run_it(TChain* tree, TString output_file)
{

  //int num_hadronic = 0;
  int correct = 0;
  int treeentries = tree->GetEntries();   
  cout << "# events in tree: "<< treeentries << endl;
  int passed_selection = 0;

  double mcwgt_intree = -999.;
  double wallTimePerEvent_intree = -99.;  
  int eventnum_intree = -999;
  int lumiBlock_intree = -999;
  int runNumber_intree = -999;
  
  vector<ttH::Jet> *preselected_jets_intree=0;
  vector<ttH::GenParticle> *gen_parts=0;
  
  tree->SetBranchStatus("*",0);
  tree->SetBranchStatus("mcwgt",1);
  tree->SetBranchStatus("wallTimePerEvent",1);
  tree->SetBranchStatus("eventnum",1);
  tree->SetBranchStatus("lumiBlock",1);
  tree->SetBranchStatus("preselected_jets.*",1);
  tree->SetBranchStatus("pruned_genParticles.*",1);


  tree->SetBranchAddress("mcwgt", &mcwgt_intree);
  tree->SetBranchAddress("wallTimePerEvent", &wallTimePerEvent_intree);
  tree->SetBranchAddress("eventnum", &eventnum_intree);
  tree->SetBranchAddress("lumiBlock", &lumiBlock_intree);
  tree->SetBranchAddress("preselected_jets", &preselected_jets_intree);
  tree->SetBranchAddress("pruned_genParticles", &gen_parts);
    
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
  
  ofstream outfile;
  outfile.open(output_file);

  Int_t cachesize = 500000000;   //500 MBytes
  tree->SetCacheSize(cachesize);
  tree->SetCacheLearnEntries(20); 


  double starttime = get_wall_time();
  //  treeentries = 1000000;
  int num_signal = 0;
  int num_bkgd = 0;
  float avg_bkgd = 0;
  for (int i=0; i<treeentries; i++)
  {

      //printProgress(i,treeentries);
      tree->GetEntry(i);
      
      cout<< "Finished: " <<(i+1)*100/treeentries <<"%\r";

      if (preselected_jets_intree->size() < 3) continue;
      vector<ttH::Jet> matched_jets[2] = {vector<ttH::Jet>(), vector<ttH::Jet>()};
      vector<int> matched_ix[2] = {vector<int>(), vector<int>()};
      vector<ttH::Jet> unmatched_jets;
      int ix = 0;
      int num_pos_lept = 0;
      int num_neg_lept = 0;
      bool tau_lept = false;
      for (const auto &pjet : *preselected_jets_intree) {
        int ID = pjet.genPdgID;
        int mID = pjet.genMotherPdgID;
        int gmID = pjet.genGrandMotherPdgID;
        if (ID == 5 && mID == 6) {
          matched_jets[0].insert(matched_jets[0].begin(), pjet);
          matched_ix[0].push_back(ix);
        } else if (ID == -5 && mID == -6) {
          matched_jets[1].insert(matched_jets[1].begin(), pjet);
          matched_ix[1].push_back(ix);
        } else if (mID == 24 && gmID == 6) {
          matched_jets[0].push_back(pjet);
          matched_ix[0].push_back(ix);
        } else if (mID == -24 && gmID == -6) {
          matched_jets[1].push_back(pjet);
          matched_ix[1].push_back(ix);
        } else {
          unmatched_jets.push_back(pjet);
        }
        ix++;
      }
      
      for (const auto &gjet : *gen_parts) {
        int ID = gjet.pdgID;
        bool mother = (gjet.mother != 9999);
        bool grandmother = (gjet.grandmother != 9999);
        int mID = mother ? (*gen_parts)[gjet.mother].pdgID : 0;
        int gmID = grandmother ? (*gen_parts)[gjet.grandmother].pdgID : 0;
        if (abs(ID) == 15) {
          tau_lept = true;
        } else if (ID >= 11 && ID <= 16 && (mID == 24 || gmID == 24)) {
          num_pos_lept++;
        } else if (ID <= -11 && ID >= -16 && (mID == -24 || gmID == -24)) {
          num_neg_lept++;
        }
      }
      
      // Make sure there's only one hadronic top
      int size = preselected_jets_intree->size();
      bool matched1 = (matched_jets[0].size() == 3);
      bool matched2 = (matched_jets[1].size() == 3);
      if ((!matched1 == !matched2) || ((num_pos_lept < 2) == (num_neg_lept < 2)) || !tau_lept)
        continue;
      
      passed_selection++;
          
      // Generate all combinations of indices via 3 for loops
      // Every iteration make a vector to compare against matched vectors if there is a fully matched triplet
      // Write the triplet to file
      vector<int> best_comb = {-1, -1, -1};
      float best_score = -99;
      for (int i=0; i < size-2; i++) {
        for (int j=i+1; j < size-1; j++) {
          for (int k=j+1; k < size; k++) {
            vector<int> comb = {i, j, k};
            vector<ttH::Jet> jet_triplet = {(*preselected_jets_intree)[i], (*preselected_jets_intree)[j], (*preselected_jets_intree)[k]};
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
            
            if (score > best_score) {
                best_score = score;
                best_comb[0] = i;
                best_comb[1] = j;
                best_comb[2] = k;
            }
          }
        }
      }
    if (matched1 && (best_comb == matched_ix[0])) {
        correct++;
    } else if (matched2 && (best_comb == matched_ix[1])) {
        correct++;
    }
  }

  double endtime = get_wall_time();
  cout << "Elapsed time: " << endtime - starttime << " seconds, " << endl;
  if (treeentries>0) cout << "an average of " << (endtime - starttime) / treeentries << " per event." << endl;
  cout << "Number that passed selection: " <<passed_selection<< endl;
  cout << "Num Sig: " <<num_signal<< endl;
  cout << "Num Bkgd: " <<num_bkgd<< endl;
  cout << "Avg Bkgd: " <<avg_bkgd<< endl;
  cout << "Accuracy: " << (float) correct / passed_selection << endl;
  
  outfile.close();
}

void evaluateBDT(TString sample="")
{

  TString output_dir = "";
  
  sample = "ttH";
  TString output_file = output_dir + sample + "_triplets.csv";
  TChain *tth_chain = new TChain("OSTwoLepAna/summaryTree");    
  DIR *dir;
  struct dirent *ent;
  dir = opendir ("/scratch365/mdrnevic/trees/testing");
  while ((ent = readdir (dir)) != NULL) {
      tth_chain->Add("/scratch365/mdrnevic/trees/testing/" + (TString) ent->d_name);
  }
  closedir (dir);

  run_it(tth_chain, output_file);

}
