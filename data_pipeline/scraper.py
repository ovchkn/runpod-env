import os
import json
import logging
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from github import Github, Repository
from tavily import TavilyClient
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import mlflow

class DataScraper:
    def __init__(self, 
                 github_token: Optional[str] = None, 
                 tavily_api_key: Optional[str] = None,
                 embedding_model: str = 'all-mpnet-base-v2'):
        """
        Initialize data scraping and processing pipeline
        
        Args:
            github_token (str, optional): GitHub API token for authenticated requests
            tavily_api_key (str, optional): Tavily API key for advanced search
            embedding_model (str): Sentence transformer model for embeddings
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize GitHub client
        self.github_client = Github(github_token) if github_token else None
        
        # Initialize Tavily search client
        self.tavily_client = TavilyClient(tavily_api_key) if tavily_api_key else None
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # MLflow tracking
        mlflow.set_experiment('infrastructure_dataset_creation')

    def search_terraform_repos(self, 
                             query: str = 'infrastructure as code', 
                             max_repos: int = 50) -> List[Repository]:
        """
        Search GitHub for Terraform repositories
        
        Args:
            query (str): Search query for repositories
            max_repos (int): Maximum number of repositories to retrieve
        
        Returns:
            List of GitHub repositories
        """
        with mlflow.start_run(run_name='terraform_repo_search'):
            try:
                repos = list(self.github_client.search_repositories(
                    query=f'{query} language:HCL stars:>100', 
                    sort='stars', 
                    order='desc'
                ))[:max_repos]
                
                mlflow.log_metric('repos_found', len(repos))
                return repos
            except Exception as e:
                self.logger.error(f"Error searching repositories: {e}")
                return []

    def extract_pr_data(self, repo: Repository, max_prs: int = 100) -> List[Dict]:
        """
        Extract pull request data from a repository
        
        Args:
            repo (Repository): GitHub repository
            max_prs (int): Maximum number of PRs to retrieve
        
        Returns:
            List of PR data dictionaries
        """
        with mlflow.start_run(run_name='pr_data_extraction'):
            pr_data = []
            try:
                for pr in repo.get_pulls(state='closed')[:max_prs]:
                    pr_info = {
                        'title': pr.title,
                        'body': pr.body,
                        'created_at': pr.created_at.isoformat(),
                        'merged': pr.merged,
                        'additions': pr.additions,
                        'deletions': pr.deletions,
                        'changed_files': pr.changed_files
                    }
                    pr_data.append(pr_info)
                
                mlflow.log_metric('prs_extracted', len(pr_data))
                return pr_data
            except Exception as e:
                self.logger.error(f"Error extracting PR data: {e}")
                return []

    def tavily_infrastructure_search(self, 
                                  query: str, 
                                  max_results: int = 20) -> List[Dict]:
        """
        Perform advanced search for infrastructure-related content
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of search results
        
        Returns:
            List of search result dictionaries
        """
        with mlflow.start_run(run_name='infrastructure_search'):
            try:
                search_results = self.tavily_client.search(
                    query=query, 
                    max_results=max_results,
                    search_depth='advanced'
                )
                
                mlflow.log_metric('search_results', len(search_results.get('results', [])))
                return search_results.get('results', [])
            except Exception as e:
                self.logger.error(f"Error performing Tavily search: {e}")
                return []

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for given texts
        
        Args:
            texts (List[str]): List of text strings
        
        Returns:
            Numpy array of embeddings
        """
        return self.embedding_model.encode(texts)

    def process_and_save_dataset(self, 
                              pr_data: List[Dict], 
                              search_results: List[Dict],
                              output_dir: str = '/workspace/datasets'):
        """
        Process and save dataset with embeddings
        
        Args:
            pr_data (List[Dict]): Pull request data
            search_results (List[Dict]): Search results
            output_dir (str): Directory to save processed datasets
        """
        with mlflow.start_run(run_name='dataset_processing'):
            # Create DataFrame from PR data
            pr_df = pd.DataFrame(pr_data)
            pr_df['pr_embeddings'] = list(self.create_embeddings(pr_df['body'].fillna('')))
            
            # Create DataFrame from search results
            search_df = pd.DataFrame(search_results)
            search_df['search_embeddings'] = list(self.create_embeddings(search_df['content']))
            
            # Save processed datasets
            os.makedirs(output_dir, exist_ok=True)
            pr_df.to_parquet(f'{output_dir}/pr_dataset.parquet')
            search_df.to_parquet(f'{output_dir}/search_dataset.parquet')
            
            mlflow.log_artifacts(output_dir, artifact_path='processed_datasets')
            
            self.logger.info(f"Datasets saved to {output_dir}")

    def run_data_pipeline(self, 
                       github_query: str = 'infrastructure as code', 
                       tavily_query: str = 'DevOps infrastructure best practices'):
        """
        Execute complete data pipeline
        
        Args:
            github_query (str): Query for GitHub repository search
            tavily_query (str): Query for Tavily search
        """
        with mlflow.start_run(run_name='full_data_pipeline'):
            # Search GitHub for repositories
            repos = self.search_terraform_repos(github_query)
            
            # Extract PR data from repositories
            all_pr_data = []
            for repo in repos:
                pr_data = self.extract_pr_data(repo)
                all_pr_data.extend(pr_data)
            
            # Perform Tavily search
            search_results = self.tavily_infrastructure_search(tavily_query)
            
            # Process and save dataset
            self.process_and_save_dataset(all_pr_data, search_results)

def main():
    # Load environment variables
    github_token = os.getenv('GITHUB_TOKEN')
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    
    # Initialize and run data pipeline
    scraper = DataScraper(
        github_token=github_token, 
        tavily_api_key=tavily_api_key
    )
    scraper.run_data_pipeline()

if __name__ == '__main__':
    main()