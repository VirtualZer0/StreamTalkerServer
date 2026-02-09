#!/usr/bin/env bun

/**
 * Docker Build Script for Stream Talker Server
 *
 * Builds and pushes Docker images with version tagging.
 * Extracts version from git branch (release/X.Y.Z) or uses --version flag.
 *
 * Usage:
 *   bun run scripts/build-docker.ts --push --tag-latest
 *   bun run scripts/build-docker.ts --version 2.0.0 --dry-run
 *   bun run scripts/build-docker.ts --registry custom.io --push
 */

import { exec, execSync } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface BuildOptions {
  version?: string;
  registry: string;
  push: boolean;
  tagLatest: boolean;
  dryRun: boolean;
}

const VERSION_REGEX = /^[0-9]+\.[0-9]+\.[0-9]+$/;

/**
 * Extract version from current git branch (release/X.Y.Z)
 */
function extractVersionFromBranch(): string | null {
  try {
    const branch = execSync('git branch --show-current', { encoding: 'utf-8' }).trim();
    const match = branch.match(/^release\/([0-9]+\.[0-9]+\.[0-9]+)$/);
    return match ? match[1] : null;
  } catch (error) {
    console.error('Failed to get current git branch:', error);
    return null;
  }
}

/**
 * Validate version format (X.Y.Z)
 */
function validateVersion(version: string): boolean {
  return VERSION_REGEX.test(version);
}

/**
 * Parse command line arguments
 */
function parseArgs(): BuildOptions {
  const args = process.argv.slice(2);
  const options: BuildOptions = {
    registry: 'virtualzer0/stream-talker-server',
    push: false,
    tagLatest: false,
    dryRun: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    switch (arg) {
      case '--version':
        options.version = args[++i];
        break;
      case '--registry':
        options.registry = args[++i];
        break;
      case '--push':
        options.push = true;
        break;
      case '--tag-latest':
        options.tagLatest = true;
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      case '--help':
        printHelp();
        process.exit(0);
      default:
        console.error(`Unknown argument: ${arg}`);
        printHelp();
        process.exit(1);
    }
  }

  return options;
}

/**
 * Print help message
 */
function printHelp(): void {
  console.log(`
Docker Build Script for Stream Talker Server

Usage:
  bun run scripts/build-docker.ts [options]

Options:
  --version <X.Y.Z>       Specify version manually (defaults to git branch)
  --registry <name>       Docker registry/image name (default: virtualzer0/stream-talker-server)
  --push                  Push images to registry after build
  --tag-latest           Also tag as 'latest'
  --dry-run              Show commands without executing
  --help                 Show this help message

Environment Variables:
  DOCKERHUB_USERNAME     DockerHub username (required for --push)
  DOCKERHUB_TOKEN        DockerHub access token (required for --push)

Examples:
  # Build and push from release branch
  bun run scripts/build-docker.ts --push --tag-latest

  # Dry run to test
  bun run scripts/build-docker.ts --dry-run --version 2.0.0

  # Build specific version without pushing
  bun run scripts/build-docker.ts --version 1.0.0
  `);
}

/**
 * Run shell command
 */
async function runCommand(command: string, dryRun: boolean): Promise<void> {
  console.log(`\n$ ${command}`);

  if (dryRun) {
    console.log('[DRY RUN - command not executed]');
    return;
  }

  try {
    const { stdout, stderr } = await execAsync(command);
    if (stdout) console.log(stdout);
    if (stderr) console.error(stderr);
  } catch (error: any) {
    console.error('Command failed:', error.message);
    throw error;
  }
}

/**
 * Login to DockerHub
 */
async function dockerLogin(dryRun: boolean): Promise<void> {
  const username = process.env.DOCKERHUB_USERNAME;
  const token = process.env.DOCKERHUB_TOKEN;

  if (!username || !token) {
    throw new Error(
      'Missing DockerHub credentials. Set DOCKERHUB_USERNAME and DOCKERHUB_TOKEN environment variables.'
    );
  }

  console.log(`\nLogging in to DockerHub as ${username}...`);

  if (dryRun) {
    console.log('[DRY RUN - login skipped]');
    return;
  }

  await runCommand(`echo ${token} | docker login -u ${username} --password-stdin`, false);
}

/**
 * Build Docker image
 */
async function buildImage(version: string, registry: string, tagLatest: boolean, dryRun: boolean): Promise<void> {
  const versionTag = `${registry}:v${version}`;
  const tags = [versionTag];

  if (tagLatest) {
    tags.push(`${registry}:latest`);
  }

  const tagArgs = tags.map(tag => `-t ${tag}`).join(' ');

  const buildCommand = `docker buildx build ${tagArgs} --build-arg INCLUDE_MODEL_06B=true --build-arg INCLUDE_MODEL_17B=true .`;

  console.log(`\n=== Building Docker Image ===`);
  console.log(`Version: ${version}`);
  console.log(`Tags: ${tags.join(', ')}`);

  await runCommand(buildCommand, dryRun);
}

/**
 * Push Docker images
 */
async function pushImages(registry: string, version: string, tagLatest: boolean, dryRun: boolean): Promise<void> {
  console.log(`\n=== Pushing Docker Images ===`);

  await runCommand(`docker push ${registry}:v${version}`, dryRun);

  if (tagLatest) {
    await runCommand(`docker push ${registry}:latest`, dryRun);
  }
}

/**
 * Main execution
 */
async function main(): Promise<void> {
  console.log('Stream Talker Server - Docker Build Script\n');

  const options = parseArgs();

  // Determine version
  let version = options.version;
  if (!version) {
    version = extractVersionFromBranch();
    if (!version) {
      console.error('Error: Could not extract version from branch name.');
      console.error('Branch must be in format: release/X.Y.Z');
      console.error('Or provide version manually with --version flag.');
      process.exit(1);
    }
    console.log(`Extracted version from branch: ${version}`);
  } else {
    console.log(`Using provided version: ${version}`);
  }

  // Validate version
  if (!validateVersion(version)) {
    console.error(`Error: Invalid version format: ${version}`);
    console.error('Version must match pattern: X.Y.Z (e.g., 1.0.0)');
    process.exit(1);
  }

  if (options.dryRun) {
    console.log('\n=== DRY RUN MODE ===');
    console.log('Commands will be shown but not executed\n');
  }

  // Login to DockerHub if pushing
  if (options.push) {
    await dockerLogin(options.dryRun);
  }

  // Build image
  await buildImage(version, options.registry, options.tagLatest, options.dryRun);

  // Push if requested
  if (options.push) {
    await pushImages(options.registry, version, options.tagLatest, options.dryRun);
  }

  console.log('\n=== Build Complete ===');
  console.log(`Version: v${version}`);
  console.log(`Registry: ${options.registry}`);

  if (options.push && !options.dryRun) {
    console.log('\nImages pushed successfully:');
    console.log(`  docker pull ${options.registry}:v${version}`);
    if (options.tagLatest) {
      console.log(`  docker pull ${options.registry}:latest`);
    }
  }
}

// Run main function
main().catch(error => {
  console.error('\nBuild failed:', error.message);
  process.exit(1);
});
