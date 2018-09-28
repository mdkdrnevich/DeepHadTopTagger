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

struct less_than_key
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

void write_csv(std::ofstream& in_file, vector<ttH::Jet> in_jets, int event_num, int sig_or_bkgd)
{
  in_file << sig_or_bkgd << ",";  

  // Sort from low b-score to high b-score
  sort(in_jets.begin(), in_jets.end(), less_than_key());
  // Write out basic information
  for (const auto & jet : in_jets)
    {
      in_file<< jet.obj.pt() <<","<< jet.obj.eta() <<","<< jet.obj.phi() <<","<< jet.obj.M() <<",";
      in_file<< jet.charge <<","<< jet.DeepCSVprobb <<","<< jet.DeepCSVprobbb <<",";
      in_file<< jet.DeepCSVprobc <<","<< jet.DeepCSVprobudsg <<",";
      in_file<< jet.qgid <<",";
      in_file<< jet.ptD <<","<< jet.axis1 <<","<< jet.mult <<",";
    }
    
  // Get three types of jets
  ttH::Jet bjet = in_jets.back();
  in_jets.pop_back();
  sort(in_jets.begin(), in_jets.end(), pt_key());
  ttH::Jet wj1 = in_jets[0];
  ttH::Jet wj2 = in_jets[1];
    
  // b-jet info
  in_file<< bjet.obj.pt() <<","<< bjet.obj.M() <<","<< (bjet.DeepCSVprobb + bjet.DeepCSVprobbb) <<",";
  in_file<< (bjet.DeepCSVprobc/(bjet.DeepCSVprobc + bjet.DeepCSVprobudsg)) <<",";
  in_file<< (bjet.DeepCSVprobc/(bjet.DeepCSVprobc + bjet.DeepCSVprobb + bjet.DeepCSVprobbb)) <<",";
  in_file<< bjet.ptD <<","<< bjet.axis1 <<","<< bjet.mult <<",";
  // high pT jet info
  in_file<< wj1.obj.pt() <<","<< wj1.obj.M() <<","<< (wj1.DeepCSVprobb + wj1.DeepCSVprobbb) <<",";
  in_file<< (wj1.DeepCSVprobc/(wj1.DeepCSVprobc + wj1.DeepCSVprobudsg)) <<",";
  in_file<< (wj1.DeepCSVprobc/(wj1.DeepCSVprobc + wj1.DeepCSVprobb + wj1.DeepCSVprobbb)) <<",";
  in_file<< wj1.ptD <<","<< wj1.axis1 <<","<< wj1.mult <<",";
  // low pT jet info
  in_file<< wj2.obj.pt() <<","<< wj2.obj.M() <<","<< (wj2.DeepCSVprobb + wj2.DeepCSVprobbb) <<",";
  in_file<< (wj2.DeepCSVprobc/(wj2.DeepCSVprobc + wj2.DeepCSVprobudsg)) <<",";
  in_file<< (wj2.DeepCSVprobc/(wj2.DeepCSVprobc + wj2.DeepCSVprobb + wj2.DeepCSVprobbb)) <<",";
  in_file<< wj2.ptD <<","<< wj2.axis1 <<","<< wj2.mult <<",";
    
  // Engineered features
  TLorentzVector *tvec1 = new TLorentzVector();
  TLorentzVector *tvec2 = new TLorentzVector();
  TLorentzVector *tvec3 = new TLorentzVector();
  tvec1->SetPtEtaPhiM(bjet.obj.pt(), bjet.obj.eta(), bjet.obj.phi(), bjet.obj.M());
  tvec2->SetPtEtaPhiM(wj1.obj.pt(), wj1.obj.eta(), wj1.obj.phi(), wj1.obj.M());
  tvec3->SetPtEtaPhiM(wj2.obj.pt(), wj2.obj.eta(), wj2.obj.phi(), wj2.obj.M());
  TLorentzVector W = *tvec2 + *tvec3;
  TLorentzVector top = *tvec1 + *tvec2 + *tvec3;
    
  // Write engineered features
  in_file<< deltaR(tvec1->Eta(), tvec1->Phi(), tvec2->Eta(), tvec2->Phi()) <<","<< (*tvec1 + *tvec2).M() <<",";
  in_file<< deltaR(tvec1->Eta(), tvec1->Phi(), tvec3->Eta(), tvec3->Phi()) <<","<< (*tvec1 + *tvec3).M() <<",";
  in_file<< deltaR(tvec2->Eta(), tvec2->Phi(), tvec3->Eta(), tvec3->Phi()) <<","<< W.M() <<",";
  in_file<< deltaR(tvec1->Eta(), tvec1->Phi(), W.Eta(), W.Phi()) <<","<< top.M();
    
  in_file<<"\n";
  in_file.flush();
  
  /* OLD ENGINEERED FEATURES
  TLorentzVector *tvec1 = new TLorentzVector();
  TLorentzVector *tvec2 = new TLorentzVector();
  TLorentzVector *tvec3 = new TLorentzVector();
  tvec1->SetPtEtaPhiM(in_jets[0].obj.pt(), in_jets[0].obj.eta(), in_jets[0].obj.phi(), in_jets[0].obj.M());
  tvec2->SetPtEtaPhiM(in_jets[1].obj.pt(), in_jets[1].obj.eta(), in_jets[1].obj.phi(), in_jets[1].obj.M());
  tvec3->SetPtEtaPhiM(in_jets[2].obj.pt(), in_jets[2].obj.eta(), in_jets[2].obj.phi(), in_jets[2].obj.M());
  TLorentzVector W = *tvec2 + *tvec3;
  TLorentzVector top = *tvec1 + *tvec2 + *tvec3;
  double top_mass = top.M();
  //double top_ptDR = top.Pt() * getDeltaR(in_jets[0], (in_jets[1].obj+in_jets[2].obj));
  double top_ptDR = 0;
  double W_mass = W.M();
  double dR = getDeltaR(in_jets[1], in_jets[2]);
  double W_ptDR = W.Pt() * dR;
  double soft_drop = min((*tvec2).Pt(), (*tvec3).Pt())/(((*tvec2).Pt() + (*tvec3).Pt()) * dR * dR);
  double q2_ptd = sqrt((*tvec2).Px()*(*tvec2).Px() + (*tvec2).Py()*(*tvec2).Py() + (*tvec2).Pz()*(*tvec2).Pz())/
                  ((*tvec2).Px() + (*tvec2).Py() + (*tvec2).Pz());
  double q3_ptd = sqrt((*tvec3).Px()*(*tvec3).Px() + (*tvec3).Py()*(*tvec3).Py() + (*tvec3).Pz()*(*tvec3).Pz())/
                  ((*tvec3).Px() + (*tvec3).Py() + (*tvec3).Pz());
  double b_q2_m = (*tvec1 + *tvec2).M();
  double b_q3_m = (*tvec1 + *tvec3).M();
  
  in_file<< top_mass <<","<< top.Pt() <<","<< top_ptDR <<","<< W_mass <<","<< W_ptDR <<","<< soft_drop;
  in_file<< ","<< q2_ptd <<","<< q3_ptd <<","<< b_q2_m <<","<< b_q3_m;*/
}

void run_it(TChain* tree, TString output_file, TString sorted_file_name, TString sample)
{

  //int num_hadronic = 0;
  int treeentries = tree->GetEntries();   
  cout << "# events in tree: "<< treeentries << endl;  

  double mcwgt_intree = -999.;
  double wallTimePerEvent_intree = -99.;  
  int eventnum_intree = -999;
  int lumiBlock_intree = -999;
  int runNumber_intree = -999;
  
  vector<ttH::Jet> *preselected_jets_intree=0;
  vector<ttH::Lepton> *leptons=0;
  
  tree->SetBranchStatus("*",0);
  tree->SetBranchStatus("mcwgt",1);
  tree->SetBranchStatus("wallTimePerEvent",1);
  tree->SetBranchStatus("eventnum",1);
  tree->SetBranchStatus("lumiBlock",1);
  tree->SetBranchStatus("preselected_jets.*",1);
  tree->SetBranchStatus("preselected_leptons.*",1);


  tree->SetBranchAddress("mcwgt", &mcwgt_intree);
  tree->SetBranchAddress("wallTimePerEvent", &wallTimePerEvent_intree);
  tree->SetBranchAddress("eventnum", &eventnum_intree);
  tree->SetBranchAddress("lumiBlock", &lumiBlock_intree);
  tree->SetBranchAddress("preselected_jets", &preselected_jets_intree);
  tree->SetBranchAddress("preselected_leptons", &leptons);

  TFile *outfile = new TFile(output_file, "RECREATE"); //"UPDATE"); // #, 'test' ) // "RECREATE");
  outfile->cd();
  
  ofstream sorted_file;

  sorted_file.open(sorted_file_name);

  Int_t tree_class = 0;
  ttH::Jet tree_jet1;
  ttH::Jet tree_jet2;
  ttH::Jet tree_jet3;
  TTree tripletree("tripleTree", "Tree of triples of jets");
  tripletree.Branch("class", &tree_class);
  tripletree.Branch("jet1", &tree_jet1);
  tripletree.Branch("jet2", &tree_jet2);
  tripletree.Branch("jet3", &tree_jet3);
  Int_t cachesize = 500000000;   //500 MBytes
  tree->SetCacheSize(cachesize);
  tree->SetCacheLearnEntries(20); 


  Int_t counter = 0;
  Int_t b_counter = 0;
  double starttime = get_wall_time();
  //  treeentries = 1000000;
  for (int i=0; i<treeentries; i++)
  {
      //printProgress(i,treeentries);
      tree->GetEntry(i);
      cout<< "Finished: " <<(i+1)*100/treeentries <<"%\r";
      
     counter = 0;
     b_counter = 0;
     for (const auto &pjet : *preselected_jets_intree) {
         if ((abs(pjet.obj.eta()) < 2.5) && (pjet.obj.Pt() > 25)) {
             counter += 1;
             if (pjet.DeepCSV > 0.2219) {
                 b_counter += 1;
             }
         }
     }
     if ((counter < 4) || (b_counter < 1)) {
         continue;
     }
      
     counter = 0;
     for (const auto &lep : *leptons) {
         if ((abs(lep.obj.eta()) < 2.4) && (lep.obj.Pt() > 35)) {
             counter += 1;
         }
     }
     if (counter != 1) {
         continue;
     }

     int size = preselected_jets_intree->size();
      //write csv files
     tree_class = 0;
     // Generate all combinations of indices via 3 for loops
     // Every iteration make a vector to compare against matched vectors if there is a fully matched triplet
     // Write the triplet to file
     for (int i=0; i < size-2; i++) {
       for (int j=i+1; j < size-1; j++) {
         for (int k=j+1; k < size; k++) {
           vector<int> comb = {i, j, k};
           vector<ttH::Jet> bkgd_jets;
           for (const auto& index : comb)
             bkgd_jets.push_back((*preselected_jets_intree)[index]);
           write_csv(sorted_file, bkgd_jets, eventnum_intree, 0);
           tree_jet1 = bkgd_jets[0];
           tree_jet2 = bkgd_jets[1];
           tree_jet3 = bkgd_jets[2];
           tripletree.Fill();
         }
       }
     }
  }

  double endtime = get_wall_time();
  cout << "Elapsed time: " << endtime - starttime << " seconds, " << endl;
  if (treeentries>0) cout << "an average of " << (endtime - starttime) / treeentries << " per event." << endl;
  
  tripletree.Write();
  outfile->Close();
  sorted_file.close();
}

void makeDataWithCuts(TString sample="")
{

  TString output_dir = "";
  sample = "electron";
  TString selection = "electron";
  TString output_file = output_dir + sample + "_DeepLearningTree" + ".root";
  TChain *tth_chain = new TChain("OSTwoLepAna/summaryTree");
    
  TString datadir = "";
  datadir = "/scratch365/mdrnevic/trees/" + sample + "/";
    
  DIR *dir;
  struct dirent *ent;
  dir = opendir (datadir);
  while ((ent = readdir (dir)) != NULL) {
      tth_chain->Add(datadir + (TString) ent->d_name);
  }
  closedir (dir);

  TString file_csv = output_dir + sample + "_TreeWithCuts.csv";
     
  run_it(tth_chain, output_file, file_csv, selection);
}
