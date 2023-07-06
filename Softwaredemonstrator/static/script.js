
import * as THREE from 'https://cdn.skypack.dev/three@0.128.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.120.1/examples/jsm/controls/OrbitControls.js'
//import { OrbitControls } from 'https://unpkg.com/three@0.146.0/examples/js/controls/OrbitControls.js'
import Stats from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/libs/stats.module.js';

import { STLLoader } from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/loaders/STLLoader.js';



const container = document.getElementById( 'myCanvas' );
//document.body.appendChild( container );

const scene = new THREE.Scene()

scene.background = new THREE.Color( 0xFFFFFF );

scene.add(new THREE.AxesHelper(5))

const light = new THREE.SpotLight()
light.position.set(20, 20, 20)
scene.add(light)

const light1= new THREE.SpotLight(0x112233)
light1.position.set(20, -20, -20)
scene.add(light1)


const camera = new THREE.PerspectiveCamera(
    15,
    1,
    0.01,
    1000
)
camera.position.z = 3
camera.position.x = 3
camera.position.y = 3

//const camera = new THREE.PerspectiveCamera( 35, window.innerWidth / window.innerHeight, 1, 15 );
//camera.position.set( 425, 2, 425 );
const cameraTarget = new THREE.Vector3( 0, 0, 0 );


const renderer = new THREE.WebGLRenderer({
    antialias:true,
    canvas: container
});

renderer.outputEncoding = THREE.sRGBEncoding
//renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setSize( 400, 400 );
//document.body.appendChild(renderer.domElement)

const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping = true


const material = new THREE.MeshPhysicalMaterial({
    color: 0x999999,
    metalness: 0.25,
    opacity: 0.1,
    transmission: 0.99,
    clearcoat: 1.0,
    clearcoatRoughness: 0.25
})

const scale=0.005
const loader = new STLLoader()
loader.load(
    stlFile,
    function (geometry) {
        const mesh = new THREE.Mesh(geometry, material)
		mesh.scale.set( scale, scale, scale );
        mesh.name = 'part'
        scene.add(mesh)
    },
    (xhr) => {
        console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
    },
    (error) => {
        console.log(error)
    }
)

// loader.load(
//     bbFile,
//     function (geometry) {
//         const material = new THREE.MeshPhongMaterial( { color: 0xff5533, specular: 0x111111, shininess: 200, opacity: 0.5, transparent: true } );

//         const mesh1 = new THREE.Mesh(geometry, material)
//         mesh1.scale.set( scale, scale, scale );
//         mesh1.name = 'bBox'
//         scene.add(mesh1)
//     },
//     (xhr) => {
//         console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
//     },
//     (error) => {
//         console.log(error)
//     }
// )

// loader.load(
//     cFile,
//     function (geometry) {
//         const material = new THREE.MeshPhongMaterial( { color: 0xff5533, specular: 0x111111, shininess: 200, opacity: 0.5, transparent: true } );

//         const mesh2 = new THREE.Mesh(geometry, material)
//         mesh2.scale.set( scale, scale, scale );
//         mesh2.name = 'cBox'
//         scene.add(mesh2)
//     },
//     (xhr) => {
//         console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
//     },
//     (error) => {
//         console.log(error)
//     }
// )



window.addEventListener('resize', onWindowResize, false)
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight
	camera.aspect = 1

    camera.updateProjectionMatrix()
    //renderer.setSize(window.innerWidth, window.innerHeight)
	renderer.setSize( 400, 400 );

    render()
}


function animate() {
    requestAnimationFrame(animate)

    controls.update()

    render()

}

function render() {
	camera.lookAt( cameraTarget );
    renderer.render(scene, camera)

    if(toggleBB.checked)
    { scene.getObjectByName('bBox').visible=true }
    else 
    { scene.getObjectByName('bBox').visible=false }

    if(toggleCB.checked)
    { scene.getObjectByName('cBox').visible=true }
    else 
    { scene.getObjectByName('cBox').visible=false }
    

}

animate()

//container.appendChild( renderer.domElement );